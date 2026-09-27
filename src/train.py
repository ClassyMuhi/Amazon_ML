"""
LightGBM training for Amazon Business Entity Resolution Challenge 2026.

Input:
    output/training/training_features.tsv

Output:
    models/lightgbm_matcher.txt
    models/lightgbm_metadata.json
    output/training/validation_predictions.tsv

Important:
- Split is performed by Source1 entity.
- The same Source1 entity never appears in both train and validation.
- Model is trained on pairwise similarity features.
- Final competition threshold will be optimized separately using
  entity-level macro F0.5.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

import lightgbm as lgb

from .config import (
    TRAINING_FEATURES_FILE,
    MODEL_DIR,
    LIGHTGBM_MODEL_FILE,
    LIGHTGBM_METADATA_FILE,
    RANDOM_SEED,
    VALIDATION_SIZE,
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_OUTPUT = (
    TRAINING_FEATURES_FILE.parent
    / "validation_predictions.tsv"
)

FEATURE_COLUMNS = [
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "core_name_levenshtein_ratio",
    "core_name_token_sort_ratio",
    "core_name_token_set_ratio",
    "name_token_jaccard",
    "address_levenshtein_ratio",
    "address_token_jaccard",
    "postal_code_match",
    "address_number_match",
    "country_match",
    "is_address_missing",
]


# ============================================================
# PRINT HELPER
# ============================================================

def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_features():

    print_section("LOADING TRAINING FEATURES")

    path = Path(TRAINING_FEATURES_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Training feature file not found:\n{path}"
        )

    print(f"Reading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "label" not in df.columns:
        raise KeyError(
            "Column 'label' is missing."
        )

    if "source1_entity_id" not in df.columns:
        raise KeyError(
            "Column 'source1_entity_id' is missing."
        )

    for column in FEATURE_COLUMNS:

        if column not in df.columns:
            raise KeyError(
                f"Required feature missing: {column}"
            )

    print()
    print("Label distribution:")

    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print_section("PREPARING TRAINING DATA")

    X = df[FEATURE_COLUMNS].copy()

    y = (
        pd.to_numeric(
            df["label"],
            errors="coerce",
        )
        .fillna(0)
        .astype(np.int8)
    )

    groups = df[
        "source1_entity_id"
    ].astype(str)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(np.float32)
        )

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Positive labels: {int(y.sum()):,}"
    )

    print(
        f"Negative labels: {int((y == 0).sum()):,}"
    )

    print(
        f"Unique Source1 entities: "
        f"{groups.nunique():,}"
    )

    return X, y, groups


# ============================================================
# GROUPED TRAIN / VALIDATION SPLIT
# ============================================================

def split_data(X, y, groups):

    print_section(
        "CREATING ENTITY-LEVEL TRAIN / VALIDATION SPLIT"
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    train_indices, val_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_indices
    ].copy()

    X_val = X.iloc[
        val_indices
    ].copy()

    y_train = y.iloc[
        train_indices
    ].copy()

    y_val = y.iloc[
        val_indices
    ].copy()

    train_groups = groups.iloc[
        train_indices
    ]

    val_groups = groups.iloc[
        val_indices
    ]

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )

    print(
        f"Training Source1 entities: "
        f"{train_groups.nunique():,}"
    )

    print(
        f"Validation Source1 entities: "
        f"{val_groups.nunique():,}"
    )

    overlap = (
        set(train_groups.unique())
        &
        set(val_groups.unique())
    )

    print(
        f"Source1 entity overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Data leakage detected: "
            "Source1 entities appear in both "
            "training and validation."
        )

    print()
    print("Training labels:")
    print(y_train.value_counts())

    print()
    print("Validation labels:")
    print(y_val.value_counts())

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        val_indices,
    )


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

def train_model(
    X_train,
    X_val,
    y_train,
    y_val,
):

    print_section(
        "TRAINING LIGHTGBM MATCHER"
    )

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    print(
        f"Training positives: "
        f"{positive_count:,}"
    )

    print(
        f"Training negatives: "
        f"{negative_count:,}"
    )

    # --------------------------------------------------------
    # We deliberately do NOT use scale_pos_weight here.
    #
    # The final competition metric is entity-level F0.5.
    # Threshold optimization later will control the
    # precision/recall tradeoff.
    # --------------------------------------------------------

    model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=2000,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=100,

        subsample=0.8,

        subsample_freq=1,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=1.0,

        random_state=RANDOM_SEED,

        n_jobs=-1,

        verbosity=-1,
    )

    print()
    print("Starting training...")
    print(
        "Early stopping: 100 rounds"
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_val,
                y_val,
            )
        ],

        eval_metric=[
            "binary_logloss",
            "auc",
        ],

        callbacks=[
            lgb.early_stopping(
                100,
                verbose=True,
            ),
            lgb.log_evaluation(
                100
            ),
        ],
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_val,
    y_val,
):

    print_section(
        "PAIR-LEVEL MODEL EVALUATION"
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    ap = average_precision_score(
        y_val,
        probabilities,
    )

    print(
        f"ROC-AUC: "
        f"{auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{ap:.6f}"
    )

    print()
    print(
        "These are pair-level diagnostics."
    )

    print(
        "Competition optimization will use "
        "entity-level macro F0.5."
    )

    return probabilities


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

def save_validation_predictions(
    df,
    val_indices,
    probabilities,
):

    print_section(
        "SAVING VALIDATION PREDICTIONS"
    )

    validation_df = df.iloc[
        val_indices
    ].copy()

    validation_df[
        "prediction_probability"
    ] = probabilities

    # --------------------------------------------------------
    # Keep only columns needed for threshold optimization.
    # --------------------------------------------------------

    output_columns = [
        "source1_entity_id",
        "target_entity_id",
        "label",
        "source",
        "prediction_probability",
    ]

    validation_df = validation_df[
        output_columns
    ]

    validation_df.to_csv(
        VALIDATION_OUTPUT,
        sep="\t",
        index=False,
    )

    print(
        f"Saved: {VALIDATION_OUTPUT}"
    )

    print(
        f"Rows: {len(validation_df):,}"
    )

    return validation_df


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_columns,
):

    print_section(
        "SAVING LIGHTGBM MODEL"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save native LightGBM model
    # --------------------------------------------------------

    model.booster_.save_model(
        str(LIGHTGBM_MODEL_FILE)
    )

    print(
        f"Model saved to:\n"
        f"{LIGHTGBM_MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "model_type":
            "LightGBM LGBMClassifier",

        "objective":
            "binary",

        "feature_columns":
            feature_columns,

        "num_features":
            len(feature_columns),

        "best_iteration":
            int(
                model.best_iteration_
            ),

        "random_seed":
            RANDOM_SEED,

        "validation_size":
            VALIDATION_SIZE,

        "metric_used_for_training":
            [
                "binary_logloss",
                "auc",
            ],

        "competition_metric":
            "macro_F0.5",

        "threshold_optimization":
            "entity-level validation",

        "model_license":
            "MIT",
    }

    with open(
        LIGHTGBM_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"Metadata saved to:\n"
        f"{LIGHTGBM_METADATA_FILE}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
    feature_columns,
):

    print_section(
        "FEATURE IMPORTANCE"
    )

    importance = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<35} "
            f"{int(row['importance']):>8}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "LIGHTGBM BUSINESS ENTITY RESOLUTION TRAINING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_training_features()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y, groups = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        y_train,
        y_val,
        val_indices,
    ) = split_data(
        X,
        y,
        groups,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        X_val,
        y_train,
        y_val,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    probabilities = evaluate_model(
        model,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    save_validation_predictions(
        df,
        val_indices,
        probabilities,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print_feature_importance(
        model,
        FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "LIGHTGBM TRAINING COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Optimize the match probability threshold "
        "using entity-level macro F0.5."
    )


if __name__ == "__main__":
    main()
