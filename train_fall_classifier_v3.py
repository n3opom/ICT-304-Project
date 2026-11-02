
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold


# Reuse Version 1 preprocessing and feature extraction.
import train_fall_classifier as training


# ----------------------------------------
# Configuration
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

OUTPUT_MODEL = MODEL_DIR / "fall_classifier_v3.joblib"
OUTPUT_CV = RESULT_DIR / "v3_cross_validation.csv"
OUTPUT_PREDICTIONS = RESULT_DIR / "v3_cv_predictions.csv"

N_FOLDS = 5
RANDOM_STATE = 42


def create_model():
    """Create the same Random Forest used in V1."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    labels, videos = training.load_labels()

    video_names = sorted(labels["video"].unique())

    video_data = {}
    video_targets = []

    # ----------------------------------------
    # Step 1: Extract features from all videos
    # ----------------------------------------

    print("\nEXTRACTING VIDEO FEATURES")
    print("=" * 50)

    for name in video_names:
        features, targets = training.extract_video(
            videos[name],
            labels,
        )

        if len(features) == 0:
            raise ValueError(
                f"No labelled windows extracted from {name}"
            )

        X_video = np.asarray(features)
        y_video = np.asarray(targets)

        video_data[name] = (X_video, y_video)

        # Used to distribute fall-containing videos
        # across the five validation folds.
        has_fall = (
            (labels["video"] == name)
            & (labels["label"] == "FALL_TRANSITION")
        ).any()

        video_targets.append(int(has_fall))

    video_targets = np.asarray(video_targets)

    print("\nDATASET SUMMARY")
    print("-" * 50)
    print("Total videos:", len(video_names))
    print("Videos containing falls:", int(video_targets.sum()))
    print(
        "Videos without falls:",
        int(len(video_targets) - video_targets.sum())
    )

    if min(
        int(video_targets.sum()),
        int(len(video_targets) - video_targets.sum())
    ) < N_FOLDS:
        raise ValueError(
            "Insufficient fall/non-fall video groups "
            "for five-fold stratification."
        )

    # ----------------------------------------
    # Step 2: Five-fold cross-validation
    # ----------------------------------------

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    prediction_records = []

    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(video_names, video_targets),
        start=1,
    ):
        train_names = [
            video_names[i] for i in train_idx
        ]

        test_names = [
            video_names[i] for i in test_idx
        ]

        X_train = np.concatenate([
            video_data[name][0]
            for name in train_names
        ])

        y_train = np.concatenate([
            video_data[name][1]
            for name in train_names
        ])

        X_test = np.concatenate([
            video_data[name][0]
            for name in test_names
        ])

        y_test = np.concatenate([
            video_data[name][1]
            for name in test_names
        ])

        # Create a fresh model for this fold.
        model = create_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                labels=[1],
                average=None,
                zero_division=0,
            )
        )

        accuracy = float(np.mean(predictions == y_test))

        fold_results.append({
            "fold": fold,
            "train_videos": len(train_names),
            "test_videos": len(test_names),
            "train_windows": len(X_train),
            "test_windows": len(X_test),
            "fall_precision": float(precision[0]),
            "fall_recall": float(recall[0]),
            "fall_f1": float(f1[0]),
            "accuracy": accuracy,
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(predictions.tolist())

        # Save window predictions for each video.
        for name in test_names:
            X_video, y_video = video_data[name]
            video_predictions = model.predict(X_video)

            for actual, predicted in zip(
                y_video, video_predictions
            ):
                prediction_records.append({
                    "fold": fold,
                    "video": name,
                    "actual": int(actual),
                    "predicted": int(predicted),
                })

        print(f"\nFOLD {fold}")
        print("-" * 50)
        print("Training videos:", len(train_names))
        print("Validation videos:", test_names)
        print(f"Fall precision: {precision[0]:.2%}")
        print(f"Fall recall: {recall[0]:.2%}")
        print(f"Fall F1-score: {f1[0]:.2%}")
        print(f"Accuracy: {accuracy:.2%}")

    # ----------------------------------------
    # Step 3: Cross-validation summary
    # ----------------------------------------

    results_df = pd.DataFrame(fold_results)

    print("\nFIVE-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 50)

    for metric in [
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "accuracy",
    ]:
        mean_value = results_df[metric].mean()
        std_value = results_df[metric].std()

        print(
            f"{metric}: "
            f"{mean_value:.2%} +/- {std_value:.2%}"
        )

    print("\nPOOLED CLASSIFICATION REPORT")
    print("-" * 50)

    print(
        classification_report(
            all_true,
            all_pred,
            labels=[0, 1],
            target_names=["NON_FALL", "FALL"],
            zero_division=0,
        )
    )

    print("\nPOOLED CONFUSION MATRIX")
    print("-" * 50)

    print(
        confusion_matrix(
            all_true,
            all_pred,
            labels=[0, 1],
        )
    )

    results_df.to_csv(
        OUTPUT_CV,
        index=False,
        float_format="%.4f",
    )

    pd.DataFrame(prediction_records).to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ----------------------------------------
    # Step 4: Train final model on ALL videos
    # ----------------------------------------

    X_all = np.concatenate([
        video_data[name][0]
        for name in video_names
    ])

    y_all = np.concatenate([
        video_data[name][1]
        for name in video_names
    ])

    final_model = create_model()
    final_model.fit(X_all, y_all)

    joblib.dump({
        "model": final_model,
        "sample_fps": training.SAMPLE_FPS,
        "window_seconds": training.WINDOW_SECONDS,
        "feature_version": 1,
        "training_video_count": len(video_names),
        "cross_validation_folds": N_FOLDS,
    }, OUTPUT_MODEL)

    print("\nFINAL V3 MODEL")
    print("=" * 50)
    print("Training videos:", len(video_names))
    print("Training windows:", len(X_all))
    print("Non-fall windows:", int(np.sum(y_all == 0)))
    print("Fall windows:", int(np.sum(y_all == 1)))

    print("\nSaved model:", OUTPUT_MODEL)
    print("Saved CV metrics:", OUTPUT_CV)
    print("Saved CV predictions:", OUTPUT_PREDICTIONS)


if __name__ == "__main__":
    main()
