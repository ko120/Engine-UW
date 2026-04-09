---
title: Winter Design Report

---

# AI Video Analytics – Winter Quarter End Report



*Team Members:* Andy Cao, Brian Ko, Miles Chin, Steven Miao, Wesley Huang

*Industry Partner:* AIWaysion – AI Vision group

*Industry Mentor:* Hung‑Min Hsu

*Faculty Mentor:* Jai Jaisimha

*Submission Date:* March 19, 2026

*Institution:* University of Washington, Department of Electrical and Computer Engineering

*Location:* Seattle, Washington

### Table of Contents

1. Introduction/Abstract
2. Teams, Roles and Responsibilities
3. Project Schedule
4. Project Success Criteria
5. System Requirements
6. Hardware/Software Design
7. Design Procedure/Methods
8. Test Design
9. Realistic Constraints / Relevant Engineering Standards
10. Project Resources/Budget
11. Industry Sponsor Comments
12. References

### Introduction/Abstract

This project addresses persistent safety concerns at the unconventional intersection of **NW 43rd St and 8th Ave NW in Ballard**, where the **Burke-Gilman Trail** intersects with local streets. The corridor serves pedestrians, cyclists, and micromobility users, and it experiences recurring conflicts and near-misses with motor vehicles because of unusual intersection geometry, ignored stop signs, and unsigned trail interruptions. The project sponsor needs empirical evidence, risk metrics, and repeatable analytics to support municipal roadway-safety decisions.

Sponsored by **AIWaysion**, the project objective is to build an **AI video analytics pipeline** that can automatically detect and track diverse road users, map trajectories into world coordinates, estimate short-horizon future motion, quantify safety metrics, and support engineering recommendations. The planned technical deliverables are: **(1)** improved computer-vision models for pedestrians, e-bikes, e-scooters, and vehicles, **(2)** trajectory-based safety metrics such as **time-to-collision (TTC)**, **post-encroachment time (PET)**, and yielding compliance, **(3)** conflict heatmaps by time of day and week, **(4)** evidence-based roadway countermeasure recommendations, **(5)** a professional final report, **(6)** a multimodal dataset of trajectories and interactions, and **(7)** a reproducible codebase for the video analytics pipeline.

The Winter quarter work completed the project through **Milestone D: Camera Calibration + world-coordinate trajectory mapping**. Specifically, the team completed project setup, data curation and annotation for an initial gold set, an end-to-end baseline computer-vision pipeline, and homography-based camera calibration for mapping image coordinates to the ground plane. Current slide-deck evidence shows completed work on dataset generation, tracking comparison, YOLO26 fine-tuning, and trajectory prediction with homography calibration. The remaining work for Spring quarter centers on safety/conflict analytics, improved micromobility detection and tracking, countermeasure analysis, and the final delivery package.

### Teams, Roles and Responsibilities

The project team comprises five students: **Andy Cao, Brian Ko, Miles Chin, Steven Miao, and Wesley Huang**, guided by **Industry Mentor Hung‑Min Hsu** and **Faculty Mentor Jai Jaisimha**. Mentors from AIWaysion (Dr. Hsu and Dr. Sun) and the university (Professor Jaisimha) offered technical guidance and project oversight.

| Team Member | Role/Responsibilities | Actual Contributions |
|---|---|---|
| **Andy Cao** | Micromobility Team / Model train | Andy contributed to the development of the object recognition and labeling pipeline. Performed a comparative analysis of detection results across different models (Sam3/YOLO) and labeling schemas. Also manually annotated to build up the basic datasets. Furthermore, fine tuned YOLO model specifically on 'easy' dataset, evaluating training outcomes under different hyperparameters to determine the most effective metrics.
| **Brian Ko** | Near-miss Team / Experiment Lead | Brian contributed to the core technical design of the project by developing the pseudo label generation pipeline, which combined automated detection and tracking outputs with human review to create higher quality training and evaluation data. He also contributed to the trajectory prediction pipeline, including baseline motion forecasting methods and evaluation using trajectory error metrics, and helped design the homography based calibration pipeline that maps image coordinates into real world ground plane coordinates for speed estimation and  trajectory analysis.  |
| **Miles Chin** | Near-miss Team/ Data annotation | Label the hard set of goldset, implement the initial version of Yolo inference pipeline, made a tool for creating new cyclist category from Yolo/Sam3 inference results, implement kalman filter of trajectory prediction |
| **Steven Miao** | Micromobility Team / Model evaluation| Steven contributed to dataset organization, annotation workflow support, and evaluation of detection results for micromobility-related objects. He also helped document project progress, prepare written report content, and support presentation development by organizing technical results and turning implementation details into clear project deliverables. |
| **Wesley Huang** | Micromobility Team / Model train| Wesley contributed to ground truth labeling with the assistance with pseudo labeling tool. Also worked project construction on CVAT, in addition to the label conversion for its compatibility in YOLO/SAM3 model. Fine-tuned YOLO model for micromobility classes on hard dataset, made evaluation metrics on adjusted hyperparameters to compare peformance, and generated demonstration video on refined results.
*Mentors:* **Hung‑Min Hsu** |Industry Mentor |Provided real‑world requirements, dataset access, and feedback on algorithms.

### Project Schedule
The project Gantt chart and link are as follows:
https://sharing.clickup.com/9017744387/g/h/8cqzq03-537/141f1d6487ab185
![未命名](https://hackmd.io/_uploads/BJPlA5o9Wx.png)

So far (Mar 20), the project has progressed through **Milestone D** and is entering the analytics and deployment phase. The milestone structure is now:

| Milestone | Tasks | Duration | Status | Evidence / Notes | Assigned Personnel |
|---|---|---|---|---|---|
| **A. Project setup** | 1. Define "minimum viable pipeline/product" (MVP). <br> 2. Create shared taxonomy (classes + conflict definitions. <br> 3. Establish engineering practice (Git, logging, dataset versioning). | 2 Weeks (Jan 5 - Jan 18) | **Completed** | Team has a defined detect → track → prediction / calibration pipeline and shared class definitions in the deck. | All members |
| **B. Data curation + annotation gold set** | 1. Select representative clips (including various video sources from different location/time/perspective).<br> 2. Build annotation guide + QA checklist. <br> 3. Label gold set + initial training set. <br> 4. Define baseline detection & tracking metrics.  | 3 Weeks (Jan 19 - Feb 8) | **Completed** | Documented the dataset snapshot, pseudo-label generation workflow, and human CVAT review. | All members |
| **C. Baseline CV pipeline running end-to-end** | 1. Ultralytics onboarding & initial training. <br> 2. Integrate detector + multi-object tracker. <br> 3. Standardize outputs (tracks, trajectories, video results). | 3 Weeks (Jan 19 - Feb 8) | **Completed** | Finished the end-to-end baseline with detection, tracking, and fine-tuning results. | All members |
| **D. Camera calibration + world-coordinate trajectory mapping** | 1. Camera calibration & homography <br> 2. Define lane / crossing geometry and ROIs <br> 3. Validate speed estimates & sanity checks. | 3 Weeks (Feb 9 - Mar 1) | **Completed** | Completed homography-based calibration and trajectory prediction results. | Devide into 2 teams: <br> **Near-miss team:** Brian Ko, Miles Chin <br> **Micro-mobility team:** Wesley Huang, Steven Miao, Andy Cao |
| **Quarterly Demo / Midpoint Review** | Consolidate phase results, prepare demo and presentation for quarterly review. | 2 Weeks (Mar 2 - Mar 22) | **Completed** | Presented and showed our production in the Winter Midpoint Presentation.| All members |
| **E. Safety/conflict analytics + repeatable reporting** | 1. Implement TTC, PET, near-miss detection, heatmaps. <br> 2. Slice analysis by time of day / day of week. <br> 3. Create “storytelling” visuals for stakeholders (City of Seattle, trail users, business owners, etc.) | 4 Weeks (Mar 23 - Apr 19) | **In progress / next** | Planned in the updated project objectives; not yet shown as completed in the deck. |
| **F. Improve detection/tracking for pedestrians and micromobility** | 1. Error analysis on conditions where baseline fails/performs poorly. <br> 2. Improve model: augmentation, re-labeling, fine-tuning. <br> 3. Re-run full analytics and show delta/improvements vs. baseline. | 3 Weeks (Apr 20 - May 10) | **In progress** | Fine-tuning results are complete for baseline datasets, but broader micromobility-specific improvement is still pending. |
| **G. Countermeasures + redesign concepts** | 1. Identify root causes from analytics for conflicts cluster, yielding noncompliance). <br> 2. Literature review. <br> 3. Propose countermeasures. | 2 Weeks (May 11 - May 24) | **Pending** | Proceed as previous stages complete. |
| **H. Final deliverable package** | Final report, poster, presentation, dataset, codebase, hand-off kit | 1 Week (May 25 - May 31) | **Pending** | Scheduled for the end of the project. |

A concise WBS based on our progress is:

1. **Project setup:** 
    (1) Define "minimum viable pipeline/product" (MVP).
    (2) Create shared taxonomy (classes + conflict definitions.
    (3) Check/estimate compute resources requirement, establish engineering practice (Git, logging, dataset versioning).
2. **Dataset and annotation:** 
    (1) Select representative clips (including various video sources from different location/time/perspective).
    (2) Establish baseline metrics, create annotation pipeline, automatically/manually label gold set and training set through CVAT.
3. **Baseline implementation:** 
    (1) Reproducible detector + tracker pipeline producing trajectories and visual outputs.
    (2) Produce annotated label data with stable tracking-id, ensure provided video has consistent quality/resolution source.
6. **Calibration and mapping:** 
    (1) **Near-miss team:** Research on Perspective-n-Point (PnP) and near-miss implementation, 2D and 3D trajectory inference for near-miss cases, near-miss evaluation on Sam3 vs YOLO26 on hard set, linear & kalman trajectory prediction.
    (2) **Micro-mobility team:** Micro-mobility detection and implementation on pose estimation, speed estimation and VLM, fine tuning on the hyperparameters of YOLO model, micro-mobility detection and evaluation on YOLO26 on baseline, easy set and hard set.
7. **Safety analytics:** 
    (1) Implement TTC, PET, near-miss detection, heatmaps.
    (2) Slice analysis by time of day / day of week.
    (3) Create “storytelling” visuals for stakeholders (City of Seattle, trail users, business owners, etc.)
9. **Model improvement:** 
    (1) Error analysis on conditions where baseline fails/performs poorly.
    (2) Improve model: augmentation, re-labeling, fine-tuning.
    (3) Re-run full analytics and show delta/improvements vs. baseline.
13. **Engineering recommendations and final hand-off:** 
    synthesize countermeasures, finalize documentation, and package the dataset and codebase.

***Compared with the earlier version of the report, the schedule can now state clearly that **Milestones A–D are complete**. Spring quarter work should therefore focus on **Milestones E–H**, especially conflict analytics, repeatable stakeholder reporting, micromobility refinement, and final delivery artifacts.***

### Project Success Criteria


| Criterion | Description | Status |
|---|---|---|
| **1. Establish a minimum viable end-to-end pipeline** | Build a reproducible pipeline that performs detection → tracking → trajectory extraction → calibration/world mapping. | **Completed** |
| **2. Create a curated gold-set dataset** | Produce representative labeled clips and an evaluation split with annotation QA. | **Completed** |
| **3. Quantify baseline detection and tracking performance** | Report metrics such as detection mAP, HOTA, gap events, and average track length. | **Completed** |
| **4. Improve detection performance through fine-tuning** | Fine-tune YOLO26x and preserve the exact measured improvements from the experiments. On the Easy set, mAP50 increased from **0.6558** to **0.9709**, precision from **0.8314** to **0.9742**, and recall from **0.8314** to **0.9329**. On the Hard set, mAP50 increased from **0.3677** to **0.8933**, precision from **0.4629** to **0.8923**, and recall from **0.3503** to **0.8418**. | **Completed** |
| **5. Map trajectories into world coordinates** | Perform camera calibration and homography-based trajectory mapping to support physically meaningful speeds, distances, and conflict zones. | **Completed** |
| **6. Quantify safety metrics** | Compute TTC, PET, yielding compliance, and other near-miss indicators from the trajectories. | **In progress** |
| **7. Produce conflict heatmaps and slice analytics** | Generate time-of-day and day-of-week risk summaries and heatmaps. | **Pending** |
| **8. Support engineering recommendations** | Translate analytics into evidence-based countermeasure proposals for City of Seattle stakeholders. | **Pending** |
| **9. Deliver the final package** | Submit the final professional report, dataset, reproducible codebase, presentation/poster, and hand-off materials. | **Pending** |

### System Requirements

The system is composed of several subsystems—detection, tracking, prediction, near‑miss analysis, micromobility classification and calibration.

 | Req. ID | Subsystem | Requirement Parameter | Test Conditions | Min Value | Nominal | Max Value |
|---|---|---|---|---|---|---|
| **SR-2** | Detection | mAP<sub>0.5:0.95</sub> on Hard dataset | Compare YOLO26x+BoT, SAM3(all), SAM3(separate) on Hard dataset | 0.1314 | 0.1407 | 0.1986 |
| **SR-3** | Tracking | HOTA on Hard dataset | Compare YOLO26x+BoT, SAM3(all), SAM3(separate) on Hard dataset | 0.2506 | 0.2682 | 0.3036 |
| **SR-4** | Prediction | ADE on Hard dataset | Compare direct estimate vs homography methods on Hard dataset | 22.175 | 23.656 | 39.658 |
| **SR-5** | Prediction | FDE on Hard dataset | Compare direct estimate vs homography methods on Hard dataset | 53.901 | 56.372 | 70.498 |



### Hardware Requirements

| ID | Requirement | Threshold |
|----|-------------|-----------|
| HR-01 | GPU for training | NVIDIA L4/H100 |
| HR-02 | Edge deployment hardware | Jetson-class GPU |
| HR-03 | Power budget | ≤ 30 W |
| HR-04 | Storage capacity | ≥ 128 GB |

### Software Requirements

| ID | Requirement |
|----|-------------|
| SR-01 | YOLO26 detection (PyTorch) |
| SR-02 | BoT-SORT tracking with optional SAM3 refinement |
| SR-03 | OpenCV homography calibration |
| SR-04 | CVAT annotation pipeline |
| SR-05 | Python-based TTC/PET analytics |

### Communication & Interface Requirements

| ID | Requirement |
|----|-------------|
| IR-01 | Accept RTSP/MP4 video input |
| IR-02 | Export trajectories in JSON/CSV |
| IR-03 | Visualization outputs (heatmaps, plots) |

### Latency Requirements

| ID | Requirement | Threshold |
|----|-------------|-----------|
| LR-01 | End-to-end inference latency | ≤ 200 ms/frame |
| LR-02 | Prediction latency | ≤ 33 ms (30 FPS) |

---

### Design Rationale

A structured decision-making process was used to compare YOLO26 + BoT-SORT and SAM3. A Pugh matrix was constructed to evaluate both pipelines across weighted criteria including accuracy, robustness, runtime, and deployability.

#### Pugh Matrix (Scored)

| Criterion | Weight | YOLO26 + BoT-SORT | SAM3 |
|---|---:|---:|---:|
| Detection accuracy (Easy) | 5 | +1 | 0 |
| Detection robustness (Hard) | 5 | +1 | -1 |
| Tracking stability | 4 | +1 | +1 |
| Runtime speed | 5 | +1 | -1 |
| Edge deployability | 5 | +1 | -1 |
| Mask quality | 3 | 0 | +1 |
| Track length | 3 | 0 | +1 |
| Annotation usefulness | 3 | +1 | +1 |
| Robustness to occlusion | 4 | +1 | -1 |

#### Weighted Score Calculation

- **YOLO26 + BoT-SORT**  
  = 5(+1) + 5(+1) + 4(+1) + 5(+1) + 5(+1) + 3(0) + 3(0) + 3(+1) + 4(+1)  
  = **31**

- **SAM3**  
  = 5(0) + 5(-1) + 4(+1) + 5(-1) + 5(-1) + 3(+1) + 3(+1) + 3(+1) + 4(-1)  
  = **-6**


### Hardware/Software Design

The system implements a **modular video analytics pipeline** for the Ballard Burke-Gilman Trail intersection study site. The architecture is intended to turn raw traffic video into trajectories, calibrated motion estimates, safety metrics, and eventually engineering recommendations.

### Code
https://github.com/ko120/Engine-Video
#### Design Pipeline

1. **Detection:** The pipeline begins with per-frame object detection. The baseline detector is **YOLO26**, which outputs bounding boxes for classes such as **car, truck, person, and bicycle**. In parallel, **SAM 3** is evaluated as an alternative segmentation-based approach that uses text prompts to generate object masks.

2. **Tracking:** Detected objects are associated across frames using **BoT-SORT** to form trajectories. For some experiments, **SAM 3** is also used to refine object identities and masks. Results show that **YOLO26 + BoT-SORT** is more stable on the **Hard** dataset, while **SAM 3** can produce longer tracks in some easier cases.

3. **Pseudo-label Generation:** A semi-automatic labeling workflow is used to create higher-quality training data. The raw videos (**Easy ID 124441** and **Hard ID 084511**) are first processed with **YOLO26 + BoT-SORT** to generate preliminary bounding boxes and track IDs. These outputs are then reviewed and corrected by humans in **CVAT**. A parallel workflow also evaluates **SAM 3**-based mask generation and identity assignment.

4. **Micromobility Classification:** The system aims to distinguish fine-grained road-user classes such as **e-bikes, scooters, wheelchairs, bicycles, and pedestrians**, since conflict severity and traffic behavior differ by category. This component depends on labeled object crops or tracks and will support more detailed downstream safety analysis.

5. **Trajectory Prediction:** Short-horizon trajectory prediction is explored using **linear extrapolation** and a **Kalman filter**. These methods estimate future motion from current tracked positions and velocities. Performance is evaluated using **Average Displacement Error (ADE)** and **Final Displacement Error (FDE)**. Experiments also compare prediction in raw image coordinates versus homography-mapped world coordinates.

6. **Near-miss Analysis:** Predicted and observed trajectories are intended to support safety analytics such as **time-to-collision (TTC)**, **post-encroachment time (PET)**, and other near-miss indicators. These outputs will be used to identify risky interactions, summarize conflict patterns, and generate stakeholder-facing safety insights. This module is planned for the next phase of the project.

7. **Calibration:** Camera calibration is performed using a **homography matrix** estimated from corresponding image and world points. This enables the conversion of pixel trajectories into approximate ground-plane coordinates for motion analysis. Experimental results indicate that selecting a smaller set of distinct calibration points produced more stable results than using a larger but noisier set of points.

8. **Edge Deployment:** The long-term deployment goal is to run the detection and tracking pipeline on an edge device using a **YOLO-based** model. This will require defining the target hardware platform, runtime constraints, and power budget during the next phase of development.

A high-level block diagram should accompany this section to show the flow of data across the major modules. The report should also explicitly list the main software frameworks used, such as **PyTorch**, **OpenCV**, and **CVAT**, and add any confirmed hardware details once available.



### UML System Architecture Diagram

The following UML‑style diagram captures the interaction between all major subsystems described above:

┌──────────────┐
│ Roadside Cam │
└───────┬──────┘
▼
┌──────────────────────┐
│ ObjectDetector       │  YOLO26 / SAM3
│ + detect(frame)      │  (PyTorch)
└───────┬──────────────┘
▼
┌──────────────────────┐
│ Tracker              │  BoT-SORT
│ + assign_ids()       │  (OpenCV)
└───────┬──────────────┘
▼
┌──────────────────────┐
│ Calibrator           │  Homography
│ + pixel_to_world()   │  (OpenCV)
└───────┬──────────────┘
▼
┌──────────────────────┐
│ Predictor            │  Linear / Kalman
│ + forecast()         │  (NumPy/SciPy)
└───────┬──────────────┘
▼
┌──────────────────────┐
│ NearMissEngine       │  TTC / PET
│ + compute_risk()     │
└───────┬──────────────┘
▼
┌──────────────────────┐
│ MicromobilityClass   │  CNN / LightGBM
│ + classify(track)    │
└──────────────────────┘

### Core Equations Used in Implementation


#### YOLO26 Loss Function
The YOLO26 architecture slide shows the standard multi‑term detection loss:

$$
L = \lambda_{box} \, L_{box} + \lambda_{obj} \, L_{obj} + \lambda_{cls} \, L_{cls}
$$

---

### Mean Average Precision (mAP)

From the detection metrics slide:

Average Precision for class \(c\):

$$
AP_c = \int_0^1 P(R) \, dR
$$

Mean Average Precision across all classes:

$$
mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c
$$

Where:  
- \(P(R ) \) is the precision–recall curve  
- \(C\) is the number of classes  

---
### HOTA (Higher Order Tracking Accuracy)

From tracking:

$$
HOTA = \frac{1}{|A|} \sum_{a \in A} \sqrt{Det_a \cdot Assoc_a}
$$

Where:  
- \(Det_a\) = detection accuracy for association threshold \(a\)  
- \(Assoc_a\) = association accuracy for threshold \(a\)  
- \(A\) = set of thresholds  

HOTA rewards both **correct detection** and **identity consistency**.

---

#### Homography Mapping
Camera Calibration: 2D to 3D Mapping

$$
p_{world} = H \cdot p_{pixel}
$$

Where:

- \( p_{pixel} = (u, v, 1)^T \)
- \( H \) is a \( 3 \times 3 \) homography matrix
- \( p_{world} = (X, Y, 1)^T \)

---

### Trajectory Prediction

#### Linear Extrapolation
Used as the baseline short‑horizon predictor:

$$
x_{t+k} = x_t + k \cdot v_x
$$

$$
y_{t+k} = y_t + k \cdot v_y
$$

#### Kalman Filter (State Update)
As described in the “Short horizon trajectory prediction” slide:

$$
\mathbf{x}_{t+1} = A \, \mathbf{x}_t + B \, \mathbf{u}_t
$$

Where the state vector is:

$$
\mathbf{x} = [x, y, v_x, v_y]^T
$$

---

### ADE & FDE (Prediction Metrics)


Average Displacement Error (ADE):

$$
ADE = \frac{1}{T} \sum_{t=1}^{T} \sqrt{(x_t - \hat{x}_t)^2 + (y_t - \hat{y}_t)^2}
$$

Final Displacement Error (FDE):

$$
FDE = \sqrt{(x_T - \hat{x}_T)^2 + (y_T - \hat{y}_T)^2}
$$

---

### Near‑Miss Metrics (TTC & PET)

#### Time‑to‑Collision (TTC)
From the near‑miss slide:

$$
TTC = \frac{d}{v_{rel}}
$$

Where:
- \( d \) = distance between two road users  
- \( v_{rel} \) = relative closing speed  

#### Post‑Encroachment Time (PET)

$$
PET = t_{exit,A} - t_{enter,B}
$$

A near‑miss is flagged when:
- \( TTC < \text{threshold} \)  
- or \( PET < \text{threshold} \)  
- or minimum distance < safety buffer 

### Design Procedure/Methods


1. **Dataset creation:** Two 3-minute crossing videos (Easy ID 124441 and Hard ID 084511) were selected. A **pseudo-label generation** pipeline was built: YOLO26 detector and BoT-SORT tracker generated initial bounding boxes and identities, which were manually corrected using CVAT. An alternative pipeline used SAM 3 prompts to generate segmentation masks and IDs.

2. **Model selection:** A **Pugh matrix or decision rationale** should justify the choice between YOLO26 + BoT-SORT and SAM 3. Current results show that YOLO26 + BoT-SORT has higher mAP and HOTA on the Hard set, whereas SAM 3 (separate prompts) yields longer tracks and fewer gaps on the Easy set. The choice for deployment remains to be decided, balancing detection accuracy and tracking consistency.

3. **Fine-tuning experiments:** Experiment 2 fine-tuned the YOLO26x detector on each dataset. Hyperparameters explored included learning rate ∈ {0.0005, 0.001, 0.01}, image size ∈ {640, 960}, patience ∈ {10}, batch size ∈ {16, 32, 64}, maximum frames ∈ {400, 600, 1000}, and maximum frames per class ∈ {100, 150, 250}. Training ran for 100 epochs on GPUs (L4, H100) with base model `yolo26x.pt` and dataset scale of 250 frames per class. Results show significant improvements: on the Easy set, mAP50 increased from 0.6558 to 0.9709 and precision improved from 0.8314 to 0.9742; on the Hard set, mAP50 improved from 0.3677 to 0.8933 and precision from 0.4629 to 0.8923.

4. **Tracking comparison:** Experiment 1 compared the baseline YOLO26 + BoT-SORT to SAM 3 variants. The results report HOTA, mAP, gap events, and average track length for each model on the Easy and Hard datasets. The main findings were that SAM 3 with separate prompts yields the highest HOTA (0.3244) and longest average track length (87.2) on the Easy set but with lower mAP, while YOLO26 + BoT-SORT achieves higher HOTA (0.3036) and mAP (0.1986) on the Hard set with fewer gaps than SAM 3. Video examples also show qualitative differences.

5. **Trajectory prediction and calibration:** Experiment 3 examined trajectory prediction algorithms with and without homography calibration [11][12]. A table summarizes ADE and FDE for linear and Kalman methods using direct pixel coordinates versus homography mapping [11][13]. Homography mapping lowers ADE/FDE for linear predictions but worsens Kalman results due to numerical instabilities when transforming back and forth through an ill-conditioned matrix. Figures show that selecting fewer but more distinct calibration points (six points) leads to more reliable homography than using eight noisy points.

6. **Future work planning:** Directions for Spring quarter include near-miss detection using trajectory predictions, adopting transformer-based trajectory prediction models to capture long-range dependencies, and deploying the detection module on edge devices using YOLO. Additional experiments, such as micromobility classification and near-miss analysis, are planned but not yet executed.

The report should include flow charts of the labeling workflow, model training pipeline, and trajectory prediction process; these were illustrated previously but need to be redrawn with proper captions and referenced in the text.

### Test Design

**Detection Testing:** Each dataset is divided into **training, validation, and test** subsets. Detection performance is evaluated on the held-out test set using **mAP<sub>0.5:0.95</sub>**. The main comparison is between the **pretrained YOLO26 model** and the **fine-tuned YOLO26 model** to measure the effect of domain-specific training.

**Tracking Testing:** Tracking performance is evaluated using **Higher Order Tracking Accuracy (HOTA)**, **gap events**, and **average track length**. The comparison includes **YOLO26 + BoT-SORT**, **SAM 3 (all prompts)**, and **SAM 3 (separate prompts)** to assess trajectory consistency and identity preservation across methods.

**Micromobility Classification Testing:** A labeled evaluation subset is used to distinguish fine-grained road-user categories such as **e-bikes, scooters, wheelchairs, bicycles, and pedestrians**. Performance should be measured using **accuracy, precision, recall, and F1-score**. Since this module is still under development, this testing procedure is planned for the next phase.

**Trajectory Prediction Testing:** Trajectory prediction performance is evaluated using **Average Displacement Error (ADE)** and **Final Displacement Error (FDE)**. The experiments compare **linear extrapolation** and **Kalman filter** methods under both **direct image-coordinate estimation** and **homography-based world-coordinate mapping**. Results are reported separately for the **Easy** and **Hard** datasets.

**Field Validation:** Full-system validation will be conducted at the target roadway site by comparing detected trajectories and identified risky interactions against **human-reviewed observations**. This validation is intended to confirm that the integrated pipeline can support practical near-miss analysis under real-world conditions. This step is scheduled for the Spring quarter.

### Realistic Constraints / Relevant Engineering Standards

* **Data privacy and ethics:** Video surveillance raises privacy concerns; compliance with data protection regulations (e.g., GDPR and local privacy laws) is essential. Policies on storing and processing personally identifiable information must be defined. [4][5]

* **Compute resources:** Training deep models (YOLO26, SAM 3) requires GPUs; the budget lists a Google Colab Pro+ subscription and mentions L4 and H100 GPUs. Edge deployment must account for limited processing power and energy consumption; hardware specifications are missing. [1][2]

* **Labeling effort:** Generating accurate labels through CVAT is labor-intensive; thus, budgets and schedules must account for human annotation time. [6]

* **Environmental conditions and site geometry:** The Ballard study site has unusual intersection angles and mixed trail/street interactions. Varying lighting, weather, occlusion, and camera placement can affect detection, tracking, calibration, and risk estimation; robustness to these factors should be considered in the system requirements. [7][8]

* **Safety standards:** If the system influences traffic control or pedestrian alerts, relevant transportation and safety standards (e.g., MUTCD in the U.S.) may apply. Identification of applicable standards is missing and should be researched. [9]

* **Regulatory compliance:** Use of wireless connectivity or edge devices may require FCC certification; data storage may need IT security compliance. [10]

### Project Resources/Budget

The Winter report guideline states that this section should include only costs that have been incurred, will require purchase orders, or require reimbursement, and that any costs covered directly by the company should be identified. Consistent with that guidance, the currently documented Winter-quarter expenses are limited to software and cloud-computing resources used directly for annotation and model development. [7][8]

The slides specify that the **CVAT** subscription cost **$35 per month** for two months, and **Google Colab Pro+** cost **$55** for one month, for a total documented Winter-quarter expenditure of **$125**. CVAT supported manual review and correction of labels, while Colab Pro+ supported model training and experimentation. [6]

Additional project resources mentioned in the report include access to **L4** and **H100** GPU environments for model training, as well as a plan to purchase or configure an edge device for later deployment. However, those hardware details, ownership, and associated costs are not yet fully specified, so they should be listed as pending rather than reported as finalized expenditures at this stage. [7][8]

A summary of the currently known budget items is shown below. Any future hardware purchases, sponsor-covered expenses, or reimbursement-based purchases should be added once confirmed. [7][8]

| Item | Cost | Notes |
|---|---|---|
| CVAT subscription | $35 × 2 months = $70 | Labeling tool used by the team |
| Google Colab Pro+ subscription | $55 × 1 month = $55 | Cloud GPU for training |
| **Total spent (winter)** | **$125** | |
| Edge device hardware | [Missing] | Planned for spring quarter |
| Additional sensors or cameras | [Missing] | Not specified |

Any purchase orders covered directly by the company and reimbursement details are not yet included and should be added once that information is available. [8]

### Industry Sponsor Comments

The team demonstrated good technical progress this quarter and successfully completed the foundational stages of the project, including dataset preparation, detection and tracking pipeline development, and camera calibration for world coordinate mapping. These efforts align well with the project objectives and provide a strong basis for the remaining work. For the next phase, the team should prioritize safety metric analysis, system validation, and the development of actionable recommendations for stakeholders.

### References

1. Carion, N., et al., *SAM 3: Segment anything with concepts*, arXiv:2511.16719, 2025.
2. Sapkota, R., et al., *YOLO26: key architectural enhancements and performance benchmarking for real-time object detection*, arXiv:2509.25164, 2025.
3. [Include other references used for Kalman filtering, homography estimation, evaluation metrics, or standards as needed].
4. European Commission, “Principles of the GDPR,” European Commission, accessed March 2026.
5. European Commission, “Data protection rules for business and organisations,” European Commission, accessed March 2026.
6. CVAT.ai Corporation, “CVAT Documentation,” CVAT Docs, accessed March 2026.
7. R. B. Darling, *Research and Development Project Report Guide*, Rev. 3, Dec. 9, 2019.
8. *Guidelines for Winter Quarter End Report*, course handout, accessed March 2026.
9. Federal Highway Administration, *Manual on Uniform Traffic Control Devices for Streets and Highways (MUTCD)*, current edition, accessed March 2026.
10. Federal Communications Commission, “Equipment Authorization,” FCC, accessed March 2026.
11. R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.
12. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2004.
13. A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, F. Li, and S. Savarese, “Social LSTM: Human Trajectory Prediction in Crowded Spaces,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 961–971.


