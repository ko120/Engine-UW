"""
Deep-learning trajectory predictor (Transformer encoder-decoder).

Trains on CVAT-annotated tracks, then exposes `transformer_predict` with the
same (history, horizon) -> list[(x, y)] signature as the linear/Kalman
predictors in `traj_predict_from_cvat.py`, so the three methods can be
compared head-to-head.

Architecture
------------
  • Input: `lookback` past centres, expressed relative to the last observed
    point and scaled so most samples live in O(1) magnitude.
  • Encoder: standard `nn.Transformer` encoder over the past tokens,
    sinusoidal positional encoding.
  • Decoder: `horizon` learned query embeddings cross-attend to the encoder
    memory and are projected back to (dx, dy); all future steps are
    predicted in parallel (DETR-style), no autoregression needed.
  • Output: (dx, dy) future displacements rescaled + re-anchored to the
    observed frame.

Usage
-----
    # Train on the easy split and print a linear/Kalman/Transformer table:
    python transformer_trajectory.py train \\
        --input  label/easy/annotations_easy.xml \\
        --model-out models/traj_transformer.pt \\
        --horizon 30 --lookback 10 --epochs 100 --compare

    # Re-evaluate an existing checkpoint:
    python transformer_trajectory.py eval \\
        --input  label/easy/annotations_easy.xml \\
        --model-in models/traj_transformer.pt
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reuse XML parsing + classical baselines from the original script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_predict_from_cvat import (
    parse_cvat_xml,
    frame_to_center,
    linear_predict,
    kalman_predict,
    group_cyclist_agents,
    build_output_xml,
)


# ── device ───────────────────────────────────────────────────────────────────

def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── model ────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── kinematic features + constant-velocity baseline ─────────────────────────
#
# Two helpers that together let the Transformer focus on the *curve correction*
# rather than rebuilding a trajectory from scratch:
#
#   • _compute_kinematic_features turns a (L, 2) pixel-track into 8 per-step
#     channels — relative position, velocity, speed, sin/cos heading, and
#     turn-rate.  Velocity + heading + turn-rate are exactly the signals the
#     model needs to detect that an object is *turning*.
#
#   • _linear_baseline forecasts a constant-velocity future from the same
#     window.  The Transformer then only has to predict the residual on top,
#     so straight motion is "free" and the network's capacity is spent on
#     curves.

INPUT_FEATURE_DIMS = {"position": 2, "kinematic": 8}


def _compute_kinematic_features(history_arr: np.ndarray, scale: float) -> np.ndarray:
    """
    history_arr : (L, 2) float32 pixel positions, oldest first.
    Returns (L, 8) channels: [rel_x, rel_y, vx, vy, speed, sin_h, cos_h, turn_rate]
    All values are scaled into O(1) magnitudes.
    """
    history_arr = np.asarray(history_arr, dtype=np.float32)
    L = history_arr.shape[0]
    origin = history_arr[-1]
    rel = ((history_arr - origin) / scale).astype(np.float32)

    if L >= 2:
        diff = history_arr[1:] - history_arr[:-1]
        diff = np.concatenate([diff[:1], diff], axis=0)  # left-pad first diff
    else:
        diff = np.zeros((L, 2), dtype=np.float32)
    vel = (diff / scale).astype(np.float32)

    speed = np.linalg.norm(vel, axis=1, keepdims=True).astype(np.float32)
    heading = np.arctan2(diff[:, 1], diff[:, 0]).astype(np.float32)
    sin_h = np.sin(heading).reshape(-1, 1).astype(np.float32)
    cos_h = np.cos(heading).reshape(-1, 1).astype(np.float32)

    if L >= 2:
        h_unwrap = np.unwrap(heading)
        turn = np.zeros(L, dtype=np.float32)
        turn[1:] = (h_unwrap[1:] - h_unwrap[:-1]).astype(np.float32)
    else:
        turn = np.zeros(L, dtype=np.float32)
    turn = turn.reshape(-1, 1).astype(np.float32)

    return np.concatenate([rel, vel, speed, sin_h, cos_h, turn], axis=1)


def _linear_baseline(history_arr: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    """
    Constant-velocity forecast for `horizon` future steps, scaled and expressed
    relative to history_arr[-1].  Mirrors the linear_predict baseline used in
    traj_predict_from_cvat so the residual the Transformer learns is exactly
    the gap between the Linear predictor and ground truth.
    """
    history_arr = np.asarray(history_arr, dtype=np.float32)
    L = history_arr.shape[0]
    if L >= 2:
        v = (history_arr[-1] - history_arr[0]) / max(L - 1, 1)
    else:
        v = np.zeros(2, dtype=np.float32)
    steps = np.arange(1, horizon + 1, dtype=np.float32).reshape(-1, 1)
    return ((steps * v) / scale).astype(np.float32)


def _compute_features(history_arr: np.ndarray, scale: float, mode: str) -> np.ndarray:
    """Dispatches to the right feature builder; legacy "position" mode keeps
    the original 2-channel relative-position representation so old checkpoints
    still load and infer correctly."""
    if mode == "position":
        history_arr = np.asarray(history_arr, dtype=np.float32)
        origin = history_arr[-1]
        return ((history_arr - origin) / scale).astype(np.float32)
    if mode == "kinematic":
        return _compute_kinematic_features(history_arr, scale)
    raise ValueError(f"Unknown feature_mode={mode!r}")


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_dim: int = 2,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max(lookback, horizon) + 4
        )

        # Learned decoder queries — one per future step.
        self.query_embed = nn.Parameter(torch.randn(horizon, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, 2)
        # Small-random init keeps the initial residual tiny (so the model
        # boots up close to the linear baseline) while still letting gradients
        # flow upstream through W^T — zero-init blocks that path entirely.
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        baseline: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        features : (B, L, F) input feature tokens (F=2 legacy, F=8 kinematic).
        baseline : (B, H, 2) constant-velocity baseline in scaled-relative
                   coords.  When provided, the network outputs a *residual*
                   that is added to the baseline (kinematic mode); when None
                   the raw output is returned (legacy / position mode).
        Returns  : (B, H, 2) future positions in scaled-relative coords.
        """
        B = features.size(0)
        src = self.pos_enc(self.input_proj(features))
        tgt = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer(src, tgt)
        residual = self.output_proj(out)
        if baseline is None:
            return residual
        return baseline + residual


# ── data ─────────────────────────────────────────────────────────────────────

def extract_pairs(tracks, lookback: int, horizon: int):
    """
    Walk every track and every frame t that has both `lookback` past frames
    and `horizon` future frames.  Returns a list of (track_id, history, future)
    tuples with shapes (L, 2) and (H, 2) in raw pixel coordinates.
    """
    pairs = []
    for track in tracks:
        tid = track["id"]
        fc = frame_to_center(track)
        fidxs = sorted(fc.keys())
        for i, t in enumerate(fidxs):
            future_fidxs = [f for f in fidxs if f > t][:horizon]
            if len(future_fidxs) < horizon:
                continue
            past_fidxs = fidxs[max(0, i - lookback + 1): i + 1]
            if len(past_fidxs) < lookback:
                continue
            history = np.asarray([fc[f] for f in past_fidxs], dtype=np.float32)
            future = np.asarray([fc[f] for f in future_fidxs], dtype=np.float32)
            pairs.append((tid, history, future))
    return pairs


class TrajectoryDataset(Dataset):
    """
    Builds (features, baseline, target) triples from raw CVAT tracks.

    • `features`  – kinematic or position feature tokens for the history.
    • `baseline`  – constant-velocity forecast over the horizon, scaled and
                    relative to the last observed point.  In legacy
                    "position" mode this is zeros (no residual learning).
    • `target`    – ground-truth future, scaled and relative to the same
                    anchor as the baseline.

    Augmentation rotates and optionally horizontally-flips the *raw* track
    around the last observed point so feature/baseline computation sees the
    augmented motion (e.g., a left turn becomes a right turn under flip).
    """

    def __init__(
        self,
        pairs,
        scale: float = 50.0,
        augment: bool = False,
        feature_mode: str = "kinematic",
        curve_aug_prob: float = 0.5,
        curve_aug_omega_max: float = 0.025,
    ):
        self.pairs = pairs
        self.scale = float(scale)
        self.augment = augment
        self.feature_mode = feature_mode
        self.curve_aug_prob = float(curve_aug_prob)
        self.curve_aug_omega_max = float(curve_aug_omega_max)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, history, future = self.pairs[idx]
        history = np.asarray(history, dtype=np.float32).copy()
        future = np.asarray(future, dtype=np.float32).copy()

        if self.augment:
            # 1. Global rotation around the anchor (history[-1]).
            theta = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            anchor = history[-1].copy()
            history = (history - anchor) @ R.T + anchor
            future = (future - anchor) @ R.T + anchor
            if np.random.random() < 0.5:
                history[:, 0] = 2 * anchor[0] - history[:, 0]
                future[:, 0] = 2 * anchor[0] - future[:, 0]

            # 2. Synthetic curve aug: rotate each timestep around the anchor
            # by an angle proportional to its time-distance from the anchor.
            # This bends an otherwise-straight track into one with constant
            # angular velocity ω, so the model sees many more "turning"
            # trajectories than the dataset alone provides.  The same ω is
            # applied to history and future, keeping the trajectory smooth
            # across the prediction boundary.
            if (
                self.curve_aug_prob > 0.0
                and np.random.random() < self.curve_aug_prob
            ):
                omega = np.random.uniform(
                    -self.curve_aug_omega_max, self.curve_aug_omega_max
                )
                anchor = history[-1].copy()
                L_h = history.shape[0]
                H_f = future.shape[0]

                hist_t = np.arange(-(L_h - 1), 1, dtype=np.float32)
                fut_t = np.arange(1, H_f + 1, dtype=np.float32)
                ang_h = omega * hist_t
                ang_f = omega * fut_t

                h_rel = history - anchor
                f_rel = future - anchor

                ch, sh = np.cos(ang_h), np.sin(ang_h)
                cf, sf = np.cos(ang_f), np.sin(ang_f)

                history = np.stack([
                    ch * h_rel[:, 0] - sh * h_rel[:, 1],
                    sh * h_rel[:, 0] + ch * h_rel[:, 1],
                ], axis=1) + anchor
                future = np.stack([
                    cf * f_rel[:, 0] - sf * f_rel[:, 1],
                    sf * f_rel[:, 0] + cf * f_rel[:, 1],
                ], axis=1) + anchor

        H = future.shape[0]
        origin = history[-1]
        feats = _compute_features(history, self.scale, self.feature_mode)
        if self.feature_mode == "kinematic":
            baseline = _linear_baseline(history, H, self.scale)
        else:
            baseline = np.zeros((H, 2), dtype=np.float32)
        target = ((future - origin) / self.scale).astype(np.float32)

        return (
            torch.from_numpy(feats),
            torch.from_numpy(baseline),
            torch.from_numpy(target),
        )


def trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vel_weight: float = 0.5,
    acc_weight: float = 0.2,
) -> torch.Tensor:
    """
    Position + velocity + acceleration smooth-L1 loss.

    Velocity term penalises wrong step-to-step motion (so the trajectory's
    speed and direction must match GT, not just its endpoints).
    Acceleration term penalises wrong second derivatives — i.e., it directly
    rewards matching the *curvature* of the GT trajectory, which is what was
    missing for turns.
    """
    pos = F.smooth_l1_loss(pred, target)
    pv = pred[:, 1:] - pred[:, :-1]
    tv = target[:, 1:] - target[:, :-1]
    vel = F.smooth_l1_loss(pv, tv)
    if pred.size(1) > 2:
        pa = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        ta = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        acc = F.smooth_l1_loss(pa, ta)
    else:
        acc = torch.zeros((), device=pred.device)
    return pos + vel_weight * vel + acc_weight * acc


def split_by_track(pairs, val_frac: float, seed: int):
    """Hold out ~val_frac of tracks for val — prevents intra-track leakage."""
    tids = sorted({p[0] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(tids)
    n_val = max(1, int(round(len(tids) * val_frac)))
    val_tids = set(tids[:n_val])
    train_pairs = [p for p in pairs if p[0] not in val_tids]
    val_pairs = [p for p in pairs if p[0] in val_tids]
    return train_pairs, val_pairs, sorted(val_tids)


# ── training ─────────────────────────────────────────────────────────────────

def train_model(
    tracks,
    lookback: int,
    horizon: int,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    scale: float = 50.0,
    augment: bool = True,
    device: torch.device = None,
    model_kwargs: dict = None,
    save_path: str = None,
    seed: int = 42,
    verbose: bool = True,
    feature_mode: str = "kinematic",
    vel_weight: float = 1.0,
    acc_weight: float = 0.5,
    curve_aug_prob: float = 0.5,
    curve_aug_omega_max: float = 0.025,
):
    """Train the Transformer on CVAT tracks.  Returns (model, meta, val_tids)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = device or auto_device()
    model_kwargs = dict(model_kwargs or {})
    if feature_mode not in INPUT_FEATURE_DIMS:
        raise ValueError(f"Unknown feature_mode={feature_mode!r}")
    model_kwargs.setdefault("input_dim", INPUT_FEATURE_DIMS[feature_mode])
    use_residual = (feature_mode == "kinematic")

    pairs = extract_pairs(tracks, lookback, horizon)
    if not pairs:
        raise ValueError(
            f"No (history, future) pairs with lookback={lookback} horizon={horizon} "
            "could be extracted.  Check your XML, lookback, or horizon."
        )

    train_pairs, val_pairs, val_tids = split_by_track(pairs, val_frac, seed)

    # Track-level split can be empty on one side with very few tracks — fall
    # back to a shuffled sample-level split in that case.
    if not train_pairs or not val_pairs:
        rng = random.Random(seed)
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac)))
        val_pairs = shuffled[:n_val]
        train_pairs = shuffled[n_val:]
        val_tids = []

    train_ds = TrajectoryDataset(
        train_pairs, scale=scale, augment=augment, feature_mode=feature_mode,
        curve_aug_prob=curve_aug_prob, curve_aug_omega_max=curve_aug_omega_max,
    )
    val_ds = TrajectoryDataset(
        val_pairs, scale=scale, augment=False, feature_mode=feature_mode,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TrajectoryTransformer(lookback=lookback, horizon=horizon, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  device: {device}")
        print(f"  feature_mode: {feature_mode}  (input_dim={model_kwargs['input_dim']})  "
              f"residual: {use_residual}")
        print(f"  loss weights: pos=1.0  vel={vel_weight}  acc={acc_weight}")
        print(f"  curve aug: prob={curve_aug_prob}  omega_max={curve_aug_omega_max} rad/frame")
        print(f"  model params: {n_params:,}")
        print(f"  train samples: {len(train_pairs)}   val samples: {len(val_pairs)}")
        if val_tids:
            print(f"  held-out track ids: {val_tids}")
        else:
            print(f"  (track-level split empty on one side — using sample shuffle)")

    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss, n_seen = 0.0, 0
        for feats, baseline, fut in train_loader:
            feats = feats.to(device)
            baseline = baseline.to(device)
            fut = fut.to(device)
            pred = model(feats, baseline if use_residual else None)
            loss = trajectory_loss(pred, fut, vel_weight=vel_weight, acc_weight=acc_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * feats.size(0)
            n_seen += feats.size(0)
        train_loss /= max(n_seen, 1)

        model.eval()
        val_loss, ade_sum, fde_sum, n_seen = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for feats, baseline, fut in val_loader:
                feats = feats.to(device)
                baseline = baseline.to(device)
                fut = fut.to(device)
                pred = model(feats, baseline if use_residual else None)
                val_loss += trajectory_loss(
                    pred, fut, vel_weight=vel_weight, acc_weight=acc_weight,
                ).item() * feats.size(0)
                dist = torch.linalg.norm(pred - fut, dim=-1) * scale
                ade_sum += dist.mean(dim=-1).sum().item()
                fde_sum += dist[:, -1].sum().item()
                n_seen += feats.size(0)
        val_loss /= max(n_seen, 1)
        val_ade = ade_sum / max(n_seen, 1)
        val_fde = fde_sum / max(n_seen, 1)

        sched.step()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        if verbose:
            print(
                f"  ep {ep+1:>3}/{epochs}  lr={sched.get_last_lr()[0]:.2e}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"val_ADE={val_ade:.2f}px  val_FDE={val_fde:.2f}px{marker}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "scale": scale,
        "feature_mode": feature_mode,
        "model_kwargs": model_kwargs,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "meta": meta}, save_path)
        if verbose:
            print(f"  saved -> {save_path}")

    return model, meta, val_tids


# ── inference ────────────────────────────────────────────────────────────────

def load_transformer(path, device: torch.device = None):
    """Load a checkpoint saved by `train_model`.  Returns (model, meta, device).

    Adds backward compat for legacy ("position") checkpoints that pre-date the
    feature_mode / input_dim fields.
    """
    device = device or auto_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = dict(ckpt["meta"])
    meta.setdefault("feature_mode", "position")
    feature_mode = meta["feature_mode"]

    model_kwargs = dict(meta.get("model_kwargs", {}))
    model_kwargs.setdefault("input_dim", INPUT_FEATURE_DIMS[feature_mode])
    meta["model_kwargs"] = model_kwargs

    model = TrajectoryTransformer(
        lookback=meta["lookback"],
        horizon=meta["horizon"],
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta, device


def _prep_inference_inputs(history, meta):
    """Pad/truncate history, then build (features, baseline, origin) arrays
    appropriate for the checkpoint's feature_mode.

    When the available history is shorter than `lookback`, pad by
    *backward-extrapolating* the first observed velocity rather than by
    replicating the first frame.  Replication produced spurious zero-velocity
    timesteps in the kinematic feature stack — values the network never saw
    in training, which inflated ADE for early-track frames at evaluation."""
    L = meta["lookback"]
    scale = meta["scale"]
    H = meta["horizon"]
    feature_mode = meta.get("feature_mode", "position")

    h = list(history)
    if len(h) < L:
        h_arr = np.asarray(h, dtype=np.float32)
        if len(h_arr) >= 2:
            v0 = h_arr[1] - h_arr[0]
        else:
            v0 = np.zeros(2, dtype=np.float32)
        pad_count = L - len(h_arr)
        # Steps -pad_count..-1 before h[0], extrapolated at the same velocity.
        offsets = np.arange(-pad_count, 0, dtype=np.float32).reshape(-1, 1)
        pad = h_arr[0] + offsets * v0
        arr = np.concatenate([pad.astype(np.float32), h_arr], axis=0)
    else:
        arr = np.asarray(h[-L:], dtype=np.float32)

    origin = arr[-1].copy()
    feats = _compute_features(arr, scale, feature_mode)
    if feature_mode == "kinematic":
        baseline = _linear_baseline(arr, H, scale)
    else:
        baseline = None
    return feats, baseline, origin


def transformer_predict(history, horizon, *, model, meta, device=None):
    """
    Mirror of `linear_predict` / `kalman_predict`.

    Parameters
    ----------
    history : list of (x, y) pixel positions, oldest first.
    horizon : int — must be ≤ the horizon the model was trained with.

    Returns
    -------
    list of (x, y) pixel positions of length `horizon`.
    """
    if horizon > meta["horizon"]:
        raise ValueError(
            f"Requested horizon={horizon} exceeds trained horizon={meta['horizon']}."
        )
    if len(history) == 0:
        return []

    device = device or next(model.parameters()).device
    scale = meta["scale"]

    feats, baseline, origin = _prep_inference_inputs(history, meta)

    with torch.no_grad():
        f = torch.from_numpy(feats).unsqueeze(0).to(device)
        b = torch.from_numpy(baseline).unsqueeze(0).to(device) if baseline is not None else None
        pred = model(f, b)[0].cpu().numpy() * scale

    return [(float(origin[0] + dx), float(origin[1] + dy)) for dx, dy in pred[:horizon]]


def _batched_transformer_predict(histories, *, model, meta, device=None):
    """Run the model on a batch of histories.  Returns (N, H, 2) preds in px."""
    device = device or next(model.parameters()).device
    scale = meta["scale"]
    feature_mode = meta.get("feature_mode", "position")

    feats_batch = []
    base_batch = []
    origins = []
    for h in histories:
        feats, baseline, origin = _prep_inference_inputs(h, meta)
        feats_batch.append(feats)
        if baseline is not None:
            base_batch.append(baseline)
        origins.append(origin)

    feats_t = torch.from_numpy(np.stack(feats_batch, axis=0)).to(device)
    if feature_mode == "kinematic":
        base_t = torch.from_numpy(np.stack(base_batch, axis=0)).to(device)
    else:
        base_t = None

    with torch.no_grad():
        pred = model(feats_t, base_t).cpu().numpy() * scale

    origins = np.asarray(origins, dtype=np.float32)[:, None, :]  # (N, 1, 2)
    return pred + origins


# ── metric comparison ───────────────────────────────────────────────────────

def evaluate_all(
    tracks,
    lookback: int,
    horizon: int,
    *,
    model=None,
    meta=None,
    kalman_process_var: float = 1.0,
    kalman_meas_var: float = 4.0,
    restrict_to_tids=None,
    device=None,
    transformer_batch_size: int = 128,
):
    """
    Compute per-track and overall ADE/FDE for linear, Kalman, and (optional)
    Transformer predictors on every eligible frame in `tracks`.

    If `restrict_to_tids` is given, metrics are computed only over those
    tracks — useful for reporting on held-out val tracks.
    """
    predictors = ["linear", "kalman"]
    if model is not None and meta is not None:
        predictors.append("transformer")

    # 1) Gather samples in a single pass so the Transformer can run batched.
    samples = []  # each: (tid, label, history, future)
    for track in tracks:
        tid = track["id"]
        if restrict_to_tids is not None and tid not in restrict_to_tids:
            continue
        fc = frame_to_center(track)
        fidxs = sorted(fc.keys())
        for i, t in enumerate(fidxs):
            future_fidxs = [f for f in fidxs if f > t][:horizon]
            if len(future_fidxs) < horizon:
                continue
            past_fidxs = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past_fidxs]
            gt = [fc[f] for f in future_fidxs]
            samples.append((tid, track["label"], history, gt))

    if not samples:
        return {}, {p: float("nan") for p in [f"{x}_{m}" for x in predictors for m in ("ade", "fde")]}, predictors

    # 2) Run the Transformer in batches for speed.
    transformer_preds = None
    if "transformer" in predictors:
        transformer_preds = []
        for start in range(0, len(samples), transformer_batch_size):
            batch = samples[start : start + transformer_batch_size]
            hist_batch = [s[2] for s in batch]
            preds = _batched_transformer_predict(
                hist_batch, model=model, meta=meta, device=device
            )  # (N, H, 2)
            transformer_preds.extend(preds.tolist())

    # 3) Accumulate metrics per-track.
    per_track = {}
    ade_acc = {p: [] for p in predictors}
    fde_acc = {p: [] for p in predictors}

    for idx, (tid, label, history, gt) in enumerate(samples):
        preds = {
            "linear": linear_predict(history, horizon),
            "kalman": kalman_predict(
                history, horizon,
                process_var=kalman_process_var, meas_var=kalman_meas_var,
            ),
        }
        if transformer_preds is not None:
            preds["transformer"] = transformer_preds[idx]

        row = per_track.setdefault(tid, {"label": label, "n": 0,
                                         **{f"{p}_ades": [] for p in predictors},
                                         **{f"{p}_fdes": [] for p in predictors}})
        row["n"] += 1
        for name, pred in preds.items():
            dists = [math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred, gt)]
            row[f"{name}_ades"].append(float(np.mean(dists)))
            row[f"{name}_fdes"].append(float(dists[-1]))
            ade_acc[name].append(float(np.mean(dists)))
            fde_acc[name].append(float(dists[-1]))

    # Reduce per-track ADE/FDE to means and drop intermediate lists.
    for tid, row in per_track.items():
        for name in predictors:
            ades = row.pop(f"{name}_ades")
            fdes = row.pop(f"{name}_fdes")
            row[f"{name}_ade"] = float(np.mean(ades)) if ades else float("nan")
            row[f"{name}_fde"] = float(np.mean(fdes)) if fdes else float("nan")

    overall = {}
    for name in predictors:
        overall[f"{name}_ade"] = float(np.mean(ade_acc[name])) if ade_acc[name] else float("nan")
        overall[f"{name}_fde"] = float(np.mean(fde_acc[name])) if fde_acc[name] else float("nan")

    return per_track, overall, predictors


def print_metrics_table(per_track, overall, predictors, title: str = ""):
    pretty = {"linear": "Linear", "kalman": "Kalman", "transformer": "Trans"}
    cols = [(p, pretty.get(p, p.title())) for p in predictors]
    col_width = 10

    header_left = f"  {'Track':>5}  {'Label':<12}  {'N':>5}"
    header_right = "".join(f"  {disp+' ADE':>{col_width}}  {disp+' FDE':>{col_width}}"
                           for _, disp in cols)
    header = header_left + header_right
    bar = "-" * len(header)

    if title:
        print(f"\n=== {title} ===")
    print(bar)
    print(header)
    print(bar)

    for tid in sorted(per_track):
        row = per_track[tid]
        line = f"  {tid:>5}  {row['label']:<12}  {row['n']:>5}"
        for name, _ in cols:
            line += f"  {row[f'{name}_ade']:>{col_width}.3f}  {row[f'{name}_fde']:>{col_width}.3f}"
        print(line)
    print(bar)

    line = f"  {'Overall':<21}  {'':>5}"
    for name, _ in cols:
        line += f"  {overall[f'{name}_ade']:>{col_width}.3f}  {overall[f'{name}_fde']:>{col_width}.3f}"
    print(line)


# ── XML output (visualizer-compatible) ──────────────────────────────────────

def write_predictions_xml(
    xml_in_path: str,
    xml_out_path: str,
    *,
    model,
    meta,
    horizon: int,
    lookback: int,
    poly_stride: int = 5,
    kalman_process_var: float = 1.0,
    kalman_meas_var: float = 4.0,
    box_threshold: float = 0.5,
    cyclist_score_thresh: float = 0.25,
    cyclist_frac_thresh: float = 0.4,
    cyclist_min_common: int = 3,
    device=None,
):
    """
    Produce a CVAT-style predictions XML containing bbox + lin_pred +
    kalman_pred + transformer_pred + gt polylines.  The file plugs directly
    into `visualize_traj_predictions.py` (which now knows how to draw the
    new `transformer_pred_*` layer).
    """
    import xml.dom.minidom
    from xml.etree.ElementTree import tostring

    print(f"Re-parsing {xml_in_path} for XML export ...")
    meta_xml, tracks = parse_cvat_xml(xml_in_path)

    agents = group_cyclist_agents(
        tracks,
        score_thresh=cyclist_score_thresh,
        frac_thresh=cyclist_frac_thresh,
        min_common_frames=cyclist_min_common,
        verbose=False,
    )
    n_cyclists = sum(1 for t in agents if t.get("label") == "cyclist")
    print(f"  grouped into {len(agents)} agents ({n_cyclists} cyclists)")

    # Closure that bakes in model/meta/device — matches the
    # `transformer_predict_fn(history, horizon) -> [(x, y), ...]` contract.
    def pred_fn(history, h):
        return transformer_predict(history, h, model=model, meta=meta, device=device)

    print(f"Building predictions XML (horizon={horizon}, lookback={lookback}, "
          f"poly_stride={poly_stride}) ...")
    xml_root = build_output_xml(
        meta_xml, tracks,
        horizon, lookback, poly_stride,
        kalman_process_var=kalman_process_var,
        kalman_meas_var=kalman_meas_var,
        box_threshold=box_threshold,
        agent_tracks=agents,
        transformer_predict_fn=pred_fn,
    )

    raw = tostring(xml_root, encoding="unicode")
    dom = xml.dom.minidom.parseString(raw)
    pretty = ('<?xml version="1.0" encoding="utf-8"?>\n'
              + dom.toprettyxml(indent="  ").split("\n", 1)[1])

    Path(xml_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(xml_out_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"  written -> {xml_out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_train(args):
    device = torch.device(args.device) if args.device else auto_device()
    print(f"Parsing {args.input} ...")
    meta_xml, tracks = parse_cvat_xml(args.input)
    print(f"  {len(tracks)} tracks  |  {meta_xml['total_frames']} frames  |  "
          f"{meta_xml['width']}x{meta_xml['height']} px")

    print(f"\nTraining Transformer (lookback={args.lookback}, horizon={args.horizon}, "
          f"feature_mode={args.feature_mode}) ...")
    model, model_meta, val_tids = train_model(
        tracks,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        scale=args.scale,
        augment=not args.no_augment,
        device=device,
        model_kwargs={
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.enc_layers,
            "num_decoder_layers": args.dec_layers,
            "dim_feedforward": args.ff_dim,
            "dropout": args.dropout,
        },
        save_path=args.model_out,
        seed=args.seed,
        feature_mode=args.feature_mode,
        vel_weight=args.vel_weight,
        acc_weight=args.acc_weight,
        curve_aug_prob=args.curve_aug_prob,
        curve_aug_omega_max=args.curve_aug_omega_max,
    )

    if args.compare:
        if val_tids:
            pt_v, ov_v, used = evaluate_all(
                tracks, args.lookback, args.horizon,
                model=model, meta=model_meta,
                kalman_process_var=args.kalman_process_var,
                kalman_meas_var=args.kalman_meas_var,
                restrict_to_tids=set(val_tids),
                device=device,
            )
            print_metrics_table(pt_v, ov_v, used,
                                title=f"Held-out val tracks (tids={val_tids})")

        pt_f, ov_f, used = evaluate_all(
            tracks, args.lookback, args.horizon,
            model=model, meta=model_meta,
            kalman_process_var=args.kalman_process_var,
            kalman_meas_var=args.kalman_meas_var,
            device=device,
        )
        print_metrics_table(pt_f, ov_f, used,
                            title="Full dataset (Transformer has seen train tracks)")

    if args.output_xml:
        print()
        write_predictions_xml(
            args.input, args.output_xml,
            model=model, meta=model_meta,
            horizon=args.horizon, lookback=args.lookback,
            poly_stride=args.poly_stride,
            kalman_process_var=args.kalman_process_var,
            kalman_meas_var=args.kalman_meas_var,
            box_threshold=args.box_threshold,
            cyclist_score_thresh=args.cyclist_score_thresh,
            cyclist_frac_thresh=args.cyclist_frac_thresh,
            cyclist_min_common=args.cyclist_min_common,
            device=device,
        )


def cmd_eval(args):
    device = torch.device(args.device) if args.device else auto_device()

    print(f"Loading checkpoint from {args.model_in} ...")
    model, model_meta, device = load_transformer(args.model_in, device=device)
    print(f"  lookback={model_meta['lookback']}  horizon={model_meta['horizon']}  "
          f"scale={model_meta['scale']}  device={device}  "
          f"feature_mode={model_meta.get('feature_mode', 'position')}")

    print(f"\nParsing {args.input} ...")
    _, tracks = parse_cvat_xml(args.input)
    print(f"  {len(tracks)} tracks")

    pt, ov, used = evaluate_all(
        tracks, model_meta["lookback"], model_meta["horizon"],
        model=model, meta=model_meta,
        kalman_process_var=args.kalman_process_var,
        kalman_meas_var=args.kalman_meas_var,
        device=device,
    )
    print_metrics_table(pt, ov, used, title="All tracks in input XML")

    if args.output_xml:
        print()
        write_predictions_xml(
            args.input, args.output_xml,
            model=model, meta=model_meta,
            horizon=model_meta["horizon"], lookback=model_meta["lookback"],
            poly_stride=args.poly_stride,
            kalman_process_var=args.kalman_process_var,
            kalman_meas_var=args.kalman_meas_var,
            box_threshold=args.box_threshold,
            cyclist_score_thresh=args.cyclist_score_thresh,
            cyclist_frac_thresh=args.cyclist_frac_thresh,
            cyclist_min_common=args.cyclist_min_common,
            device=device,
        )


def _add_common_args(sub):
    sub.add_argument("--input", required=True, help="CVAT annotations XML")
    sub.add_argument("--kalman-process-var", type=float, default=1.0)
    sub.add_argument("--kalman-meas-var", type=float, default=4.0)
    sub.add_argument("--device", default=None,
                     help="cuda | mps | cpu (auto-detected if omitted)")

    # XML output — produces a predictions.xml with bbox + lin_pred + kalman_pred
    # + transformer_pred + gt polylines for visualize_traj_predictions.py
    sub.add_argument("--output-xml", default=None,
                     help="If given, write a full predictions XML (with transformer "
                          "polylines) to this path for visualize_traj_predictions.py.")
    sub.add_argument("--poly-stride", type=int, default=5,
                     help="Polyline keyframe every N frames in the output XML (default: 5).")
    sub.add_argument("--box-threshold", type=float, default=0.5,
                     help="Min pixel change for a new bbox keyframe (default: 0.5).")
    sub.add_argument("--cyclist-score-thresh", type=float, default=0.25)
    sub.add_argument("--cyclist-frac-thresh", type=float, default=0.4)
    sub.add_argument("--cyclist-min-common", type=int, default=3)


def main():
    ap = argparse.ArgumentParser(
        description="Transformer trajectory predictor with linear/Kalman comparison."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_train = sub.add_parser("train", help="Train the Transformer on a CVAT XML.")
    _add_common_args(ap_train)
    ap_train.add_argument("--model-out", default="models/traj_transformer.pt")
    ap_train.add_argument("--lookback", type=int, default=10)
    ap_train.add_argument("--horizon", type=int, default=30)
    ap_train.add_argument("--epochs", type=int, default=100)
    ap_train.add_argument("--batch-size", type=int, default=64)
    ap_train.add_argument("--lr", type=float, default=1e-3)
    ap_train.add_argument("--weight-decay", type=float, default=1e-4)
    ap_train.add_argument("--val-frac", type=float, default=0.2)
    ap_train.add_argument("--scale", type=float, default=50.0,
                          help="Normalization scale (px) for relative coords")
    ap_train.add_argument("--no-augment", action="store_true",
                          help="Disable rotation + horizontal flip augmentation")
    ap_train.add_argument("--seed", type=int, default=42)
    ap_train.add_argument("--d-model", type=int, default=128)
    ap_train.add_argument("--nhead", type=int, default=8)
    ap_train.add_argument("--enc-layers", type=int, default=3)
    ap_train.add_argument("--dec-layers", type=int, default=3)
    ap_train.add_argument("--ff-dim", type=int, default=256)
    ap_train.add_argument("--dropout", type=float, default=0.1)
    ap_train.add_argument("--feature-mode", choices=list(INPUT_FEATURE_DIMS.keys()),
                          default="kinematic",
                          help="Input feature set. 'kinematic' (default) feeds position + "
                               "velocity + heading + turn-rate and predicts a residual over "
                               "a constant-velocity baseline; 'position' is the legacy mode.")
    ap_train.add_argument("--vel-weight", type=float, default=1.0,
                          help="Weight on the velocity-error term in the loss (default: 1.0).")
    ap_train.add_argument("--acc-weight", type=float, default=0.5,
                          help="Weight on the acceleration/curvature-error term (default: 0.5).")
    ap_train.add_argument("--curve-aug-prob", type=float, default=0.5,
                          help="Probability of applying synthetic-curve augmentation per "
                               "training sample (default: 0.5).  Set to 0 to disable.")
    ap_train.add_argument("--curve-aug-omega-max", type=float, default=0.025,
                          help="Max |angular velocity| (rad/frame) used when synthesising "
                               "curving trajectories (default: 0.025 ≈ 1.4°/frame).")
    ap_train.add_argument("--compare", action="store_true",
                          help="After training, print ADE/FDE for linear/Kalman/Transformer.")
    ap_train.set_defaults(func=cmd_train)

    # eval
    ap_eval = sub.add_parser("eval", help="Evaluate a saved model vs linear/Kalman.")
    _add_common_args(ap_eval)
    ap_eval.add_argument("--model-in", required=True, help="Path to saved checkpoint")
    ap_eval.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
