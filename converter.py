import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime

INPUT_JSON  = '/home/brianko/Visual-Preference/test2/2_09_084511_sam3.json'
OUTPUT_XML  = '/home/brianko/Visual-Preference/test2/2_09_084511_sam3_cvat.xml'
WIDTH       = 1920
HEIGHT      = 1080
# SAM3 assigns class IDs by text prompt order (0-indexed).
# Map SAM3 sequential IDs -> canonical COCO IDs used by YOLO.
# Adjust this to match the order you passed to --text in sam3.py
# e.g. --text person bicycle car truck  ->  0=person,1=bicycle,2=car,3=truck
SAM3_TO_COCO = {0: 0, 1: 2, 2: 7, 3: 1}   # prompt order: person car truck bicycle
CLASS_NAMES  = {0: 'person', 1: 'bicycle', 2: 'car', 7: 'truck'}

print("Loading JSON...")
with open(INPUT_JSON) as f:
    data = json.load(f)

total_frames = max(f['frame_idx'] for f in data) + 1
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f+00:00')

# --- Group detections by track ID ---
# tracks[track_id] = list of (frame_idx, x1, y1, x2, y2, score, class_id)
tracks = defaultdict(list)
no_id_track = -1  # fallback counter for detections without IDs

for frame_data in data:
    frame_idx = frame_data['frame_idx']
    boxes     = frame_data['boxes_xyxy']
    scores    = frame_data['scores']
    classes     = frame_data['classes']
    class_names = frame_data.get('class_names', [])   # preferred: name-based lookup
    ids         = frame_data.get('ids', [])
    seen_track_frames = set()   # deduplicate (track_id, frame_idx) within this frame

    # Build reverse lookup: COCO name -> COCO class id
    NAME_TO_CLS = {v: k for k, v in CLASS_NAMES.items()}

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i] if i < len(scores) else 0.0

        if i < len(class_names) and class_names[i] in NAME_TO_CLS:
            # Use saved class name directly — no index mapping needed
            class_id = NAME_TO_CLS[class_names[i]]
        else:
            # Fall back to SAM3_TO_COCO index remapping
            class_id = SAM3_TO_COCO.get(int(classes[i]) if i < len(classes) else 0, -1)

        if class_id not in CLASS_NAMES:
            continue
        if i < len(ids) and ids[i] != -1:
            track_id = int(ids[i])
        else:
            track_id = no_id_track
            no_id_track -= 1
        if (track_id, frame_idx) in seen_track_frames:
            continue
        seen_track_frames.add((track_id, frame_idx))
        tracks[track_id].append((frame_idx, x1, y1, x2, y2, score, class_id))

print(f"Found {len(tracks)} unique tracks across {total_frames} frames")

# --- Build XML ---
annotations = ET.Element('annotations')
ET.SubElement(annotations, 'version').text = '1.1'

meta = ET.SubElement(annotations, 'meta')
task = ET.SubElement(meta, 'task')
ET.SubElement(task, 'id').text = '1'
ET.SubElement(task, 'name').text = 'sam3_124441_10_13'
ET.SubElement(task, 'size').text = str(total_frames)
ET.SubElement(task, 'mode').text = 'interpolation'
ET.SubElement(task, 'overlap').text = '0'
ET.SubElement(task, 'bugtracker').text = ''
ET.SubElement(task, 'created').text = now
ET.SubElement(task, 'updated').text = now
ET.SubElement(task, 'start_frame').text = '0'
ET.SubElement(task, 'stop_frame').text = str(total_frames - 1)
ET.SubElement(task, 'frame_filter').text = ''
ET.SubElement(task, 'z_order').text = 'False'

labels_el = ET.SubElement(task, 'labels')
colors = {0: '#ff6037', 1: '#0080ff'}
for class_id, class_name in CLASS_NAMES.items():
    label = ET.SubElement(labels_el, 'label')
    ET.SubElement(label, 'name').text = class_name
    ET.SubElement(label, 'color').text = colors.get(class_id, '#00ff00')
    ET.SubElement(label, 'attributes')

segments = ET.SubElement(task, 'segments')
segment  = ET.SubElement(segments, 'segment')
ET.SubElement(segment, 'id').text = '1'
ET.SubElement(segment, 'start').text = '0'
ET.SubElement(segment, 'stop').text = str(total_frames - 1)
ET.SubElement(segment, 'url').text = ''

owner = ET.SubElement(task, 'owner')
ET.SubElement(owner, 'username').text = ''
ET.SubElement(owner, 'email').text = ''

original_size = ET.SubElement(task, 'original_size')
ET.SubElement(original_size, 'width').text = str(WIDTH)
ET.SubElement(original_size, 'height').text = str(HEIGHT)
ET.SubElement(meta, 'dumped').text = now

# --- One <track> per unique track ID ---
for cvat_id, (track_id, entries) in enumerate(sorted(tracks.items())):
    entries = sorted(entries, key=lambda x: x[0])  # sort by frame_idx

    # Use class of the most common label in this track
    class_id   = Counter(e[6] for e in entries).most_common(1)[0][0]
    label_name = CLASS_NAMES.get(class_id, f'class_{class_id}')

    track_el = ET.SubElement(annotations, 'track')
    track_el.set('id', str(cvat_id))
    track_el.set('label', label_name)
    track_el.set('source', 'auto')
    track_el.set('z_order', '0')

    frames_in_track = [e[0] for e in entries]

    for idx, (frame_idx, x1, y1, x2, y2, score, _) in enumerate(entries):
        # Visible keyframe
        b = ET.SubElement(track_el, 'box')
        b.set('frame', str(frame_idx))
        b.set('outside', '0')
        b.set('occluded', '0')
        b.set('keyframe', '1')
        b.set('xtl', f'{x1:.2f}')
        b.set('ytl', f'{y1:.2f}')
        b.set('xbr', f'{x2:.2f}')
        b.set('ybr', f'{y2:.2f}')
        b.set('z_order', '0')

        # Add outside marker if there's a gap to next frame or it's the last entry
        next_frame = frames_in_track[idx + 1] if idx + 1 < len(frames_in_track) else None
        if next_frame is None or next_frame > frame_idx + 1:
            outside_frame = frame_idx + 1
            if outside_frame >= total_frames:
                continue   # last frame of video — no room for outside marker
            b2 = ET.SubElement(track_el, 'box')
            b2.set('frame', str(outside_frame))
            b2.set('outside', '1')
            b2.set('occluded', '0')
            b2.set('keyframe', '1')
            b2.set('xtl', f'{x1:.2f}')
            b2.set('ytl', f'{y1:.2f}')
            b2.set('xbr', f'{x2:.2f}')
            b2.set('ybr', f'{y2:.2f}')
            b2.set('z_order', '0')

print("Writing XML...")
tree = ET.ElementTree(annotations)
ET.indent(tree, space='  ')

with open(OUTPUT_XML, 'wb') as f:
    f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
    tree.write(f, encoding='utf-8', xml_declaration=False)

print(f"Done! Saved to: {OUTPUT_XML}")
print(f"Total CVAT tracks: {len(tracks)}")
