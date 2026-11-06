# ICT304 Intelligent Home Care System

## Random Forest Classifier Training and Evaluation

This branch, `feature/classifier-training-evaluation`, contains the training and offline evaluation code for three Random Forest fall classifiers developed for the ICT304 AI System Design project at Murdoch University (Kaplan Singapore).

**This is not the live webcam application.** To run the final camera-based V3 system, use the `main` or `feature/final-implementation` branch instead.

## Scope

- **V1:** Random Forest using 66 temporal features and a video-level training/test holdout.
- **V2:** Random Forest using 102 temporal features, including additional signed-velocity and acceleration measurements, and modified post-fall labelling.
- **V3:** Random Forest using the original 66 features, evaluated by five-fold video-level cross-validation and subsequently trained on all available labelled recordings for deployment.
- **Offline alert evaluation:** Replay annotated recordings, apply the classifier and temporal alert rules, and measure detected fall events and false alert episodes.

The scripts use MediaPipe Pose to obtain body landmarks, apply three-frame median landmark filtering, and calculate measurements of posture and movement. They operate on saved MP4 recordings rather than a live camera feed.

## Privacy and availability of recordings

**The original participant/test-subject video recordings are intentionally not published on GitHub to protect their privacy.** They are submitted separately and privately to Murdoch University through its Learning Management System (LMS) for academic assessment.

The repository does include `ICT304_Project_GroundTruthTable.xlsx`, the annotation workbook from the original project. The workbook alone is not sufficient to rerun the original experiments: the corresponding MP4 files are not publicly distributed.

**Anyone downloading this branch to run their own experiments must:**

1. Create a local `training_videos/` folder at the repository root.
2. Record or obtain their **own** videos with appropriate participant consent and place the MP4 files in that folder.
3. Create and label **their own ground-truth Excel workbook** for those videos. Save the working workbook as `ICT304_Project_GroundTruthTable.xlsx` at the repository root, using the required format below. Make a backup of the provided workbook first if you want to retain the original annotations.
4. Keep any participant recordings and sensitive annotations private. Do not upload them to the public GitHub repository.

The original dataset's metrics are historical experimental results, not guaranteed outcomes when training on a different set of recordings.

## Repository structure

```text
ICT-304-Project/
├── train_fall_classifier.py
├── train_fall_classifier_v2.py
├── train_fall_classifier_v3.py
├── evaluate_fall_alerts.py
├── evaluate_fall_alerts_v2.py
├── evaluate_fall_alerts_v3.py
├── ICT304_Project_GroundTruthTable.xlsx
├── training_videos/                  # Create locally; private MP4 recordings
├── models/
│   ├── pose_landmarker_full.task
│   ├── fall_classifier.joblib
│   ├── fall_classifier_v2.joblib
│   └── fall_classifier_v3.joblib
├── results/
│   ├── v3_cross_validation.csv
│   ├── v3_cv_predictions.csv
│   ├── ml_replay_predictions.csv
│   ├── ml_potential_alerts.csv
│   └── ml_false_alerts.csv
├── requirements.txt
└── README.md
```

The `training_videos/` folder is not included in the downloaded repository. The `models/` directory includes previously saved classifiers and the MediaPipe Pose model. The `results/` directory contains archived CSV outputs; running evaluations may overwrite them.

## Installation

Use a Python environment compatible with the versions specified in `requirements.txt`. The following example uses Windows PowerShell from the repository root:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, you can use the environment's Python executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The scripts require the MediaPipe model file at `models/pose_landmarker_full.task`, which is included in this branch. They also require local MP4 videos and a matching Excel annotation workbook to perform training or video replay.

## Preparing your own ground truth

The scripts read the **first worksheet** of `ICT304_Project_GroundTruthTable.xlsx` using `pandas.read_excel()` and require these column names:

| Column | Meaning |
| --- | --- |
| `video` | Exact MP4 filename, including the `.mp4` extension |
| `start_sec` | Start time of an annotated activity interval in seconds |
| `end_sec` | End time of that interval in seconds |
| `label` | Activity label from the supported set |

For example, annotations for a recording named `example_recording.mp4` could have consecutive intervals such as:

| video | start_sec | end_sec | label |
| --- | ---: | ---: | --- |
| example_recording.mp4 | 0.0 | 2.0 | WALKING |
| example_recording.mp4 | 2.0 | 3.0 | FALL_TRANSITION |
| example_recording.mp4 | 3.0 | 5.0 | POST_FALL |

These rows are **illustrative only** and are not part of the project dataset.

Supported labels are `FALL_TRANSITION`, `POST_FALL`, `NORMAL`, `STANDING`, `WALKING`, `SITTING`, `CROUCHING`, `CONTROLLED_LYING`, `LYING_NO_FALL`, `RECOVERY`, `NON_FALL`, and `BENDING`.

The video filename must exactly match a file in `training_videos/`. Start times must be non-negative; end times must be later than start times; intervals for the same video must not overlap; and annotations must not extend beyond the recording's duration. The training code validates these conditions before processing.

V1 and V3 treat the beginning of a `POST_FALL` interval as fall-positive for up to 0.6 seconds, then treat subsequent observations as non-fall. V2 excludes later `POST_FALL` observations from training.

## Training classifiers

From the project root, use these commands to train the models using your own recordings and matching annotations:

| Model | Command | Saved model |
| --- | --- | --- |
| V1 | `py train_fall_classifier.py` | `models/fall_classifier.joblib` |
| V2 | `py train_fall_classifier_v2.py` | `models/fall_classifier_v2.joblib` |
| V3 | `py train_fall_classifier_v3.py` | `models/fall_classifier_v3.joblib` |

V1 and V2 use a video-level holdout split (`GroupShuffleSplit`, 25% test videos). V3 uses `StratifiedKFold` with five video-level folds, saving fold metrics to `results/v3_cross_validation.csv` and held-out predictions to `results/v3_cv_predictions.csv`. V3 then trains a separate deployment model on all available labelled videos.

The V3 cross-validation procedure requires at least five recordings containing a `FALL_TRANSITION` annotation and at least five recordings without one. The original research dataset had 30 videos, but a new dataset may have a different size provided it meets the script's requirements.

Training a model overwrites its corresponding `.joblib` output. Back up supplied models before retraining if you want to preserve them.

## Offline alert evaluation

Run the evaluators from the repository root:

```powershell
py evaluate_fall_alerts.py
py evaluate_fall_alerts_v2.py
py evaluate_fall_alerts_v3.py
```

- **V1 and V2:** Load their corresponding saved `.joblib` models, then replay an explicitly defined list of eight held-out test videos. The filenames are hard-coded in `TEST_VIDEOS` near the top of each evaluator. For your own dataset, update these filenames to the appropriate **held-out** videos; do not describe the result as an independent test if those videos were used for training.
- **V3:** Recreates the five video-level folds, trains a separate model for each fold using only its training videos, and replays its held-out videos. The event evaluation is distinct from simply running the final model trained on all recordings.

The replay evaluators apply a fall probability threshold of `0.50`, require **two positive predictions within 0.5 seconds**, and apply an **eight-second detector cooldown**. A generated alert counts as a fall-event detection when it occurs during the labelled fall transition or within three seconds after its end.

The evaluators report detected and missed fall events, false alert episodes, fall-event recall, and detection delay for matched events. Classification evaluation in the training scripts separately reports metrics such as precision, recall, F1-score and accuracy.

## Results and important reproducibility notes

| File | Contents |
| --- | --- |
| `results/v3_cross_validation.csv` | V3 per-fold classification metrics |
| `results/v3_cv_predictions.csv` | V3 held-out classification predictions |
| `results/ml_replay_predictions.csv` | Time-stamped replay predictions |
| `results/ml_potential_alerts.csv` | Generated candidate alert events |
| `results/ml_false_alerts.csv` | Alerts not matched to labelled fall events |

All three replay evaluators write to the same three `ml_*.csv` paths. **Running one evaluator can overwrite the previous evaluator's replay outputs.** Save copies elsewhere before comparing multiple runs.

The archived CSV files and saved models are included to document the original development work. Without the withheld original videos, external users cannot independently reproduce the original video-based training and event replay results. They can, however, use the scripts with their own appropriately labelled dataset.

V1/V2 single-holdout scores should not be treated as directly comparable to V3 five-fold means without accounting for the different evaluation procedures. A successful fall event may also be detected even when not every temporal window during that fall is classified correctly.

## Related branches

- `main` and `feature/final-implementation`: the final live V3 webcam application.
- `feature/random-forest-integration`: the broader development archive.
- `feature/classifier-training-evaluation`: this branch; training scripts, offline evaluation, saved models, and archived results.

## Responsible use

This project is an academic research prototype, not a certified emergency-response system. Its predictions may contain false alerts or missed falls. Do not rely on it as the sole means of monitoring a person at risk of injury, and obtain appropriate consent before recording anyone.
