"""
Memory-safe training feature generation.

Reads:
    output/training/training_pairs.tsv

Loads only the Source1 and target records required by the
training candidate pairs.

Generates pairwise similarity features using src.features.

Output:
    output/training/training_features.tsv
"""

from pathlib import Path

import pandas as pd

from src.config import (
    TRAIN_DIR,
    TRAINING_PAIRS_FILE,
    TRAINING_CANDIDATES_DIR,
    TARGET_CHUNK_SIZE,
)

from src.features import compute_pair_features
from src.normalization import add_normalized_features


# ============================================================
# Dataset paths
# ============================================================

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"

OUTPUT_FILE = (
    TRAINING_CANDIDATES_DIR
    / "training_features.tsv"
)

CHUNK_SIZE = 50_000


# ============================================================
# Load training pairs
# ============================================================

def load_training_pairs():

    print("=" * 60)
    print("LOADING TRAINING PAIRS")
    print("=" * 60)

    df = pd.read_csv(
        TRAINING_PAIRS_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    df["label"] = df["label"].astype("int8")

    print(
        f"Training pairs: {len(df):,}"
    )

    print(
        f"Positive pairs: "
        f"{(df['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(df['label'] == 0).sum():,}"
    )

    return df


# ============================================================
# Load required Source1 records
# ============================================================

def load_required_source1(training_pairs):

    print()
    print("=" * 60)
    print("LOADING REQUIRED SOURCE1 RECORDS")
    print("=" * 60)

    required_ids = set(
        training_pairs[
            "source1_entity_id"
        ]
    )

    print(
        f"Required Source1 IDs: "
        f"{len(required_ids):,}"
    )

    chunks = []
    found_ids = set()

    for chunk in pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=TARGET_CHUNK_SIZE,
    ):

        matched = chunk[
            chunk["entity_id"].isin(
                required_ids
            )
        ]

        if not matched.empty:
            chunks.append(matched)
            found_ids.update(matched["entity_id"])
            if len(found_ids) >= len(required_ids):
                break

    if not chunks:

        raise RuntimeError(
            "No Source1 records found."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["entity_id"]
    )

    print(
        f"Loaded Source1 records: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# Load required target records
# ============================================================

def load_required_target_records(
    training_pairs,
    source,
):

    print()
    print("=" * 60)

    print(
        f"LOADING REQUIRED {source} RECORDS"
    )

    print("=" * 60)

    required_ids = set(
        training_pairs.loc[
            training_pairs["source"]
            == source,
            "target_entity_id",
        ]
    )

    print(
        f"Required {source} IDs: "
        f"{len(required_ids):,}"
    )

    if source == "S2":

        filepath = S2_FILE

    elif source == "S3":

        filepath = S3_FILE

    else:

        raise ValueError(
            f"Unknown source: {source}"
        )

    chunks = []
    found_ids = set()

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=TARGET_CHUNK_SIZE,
    ):

        matched = chunk[
            chunk["entity_id"].isin(
                required_ids
            )
        ]

        if not matched.empty:
            chunks.append(matched)
            found_ids.update(matched["entity_id"])
            if len(found_ids) >= len(required_ids):
                break

    if not chunks:

        raise RuntimeError(
            f"No {source} records found."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["entity_id"]
    )

    print(
        f"Loaded {source} records: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# Normalize records
# ============================================================

def normalize_records(
    s1,
    s2,
    s3,
):

    print()
    print("=" * 60)
    print("NORMALIZING RECORDS")
    print("=" * 60)

    print("Normalizing Source1...")

    s1_norm = add_normalized_features(
        s1
    )

    print("Normalizing Source2...")

    s2_norm = add_normalized_features(
        s2
    )

    print("Normalizing Source3...")

    s3_norm = add_normalized_features(
        s3
    )

    return (
        s1_norm,
        s2_norm,
        s3_norm,
    )


# ============================================================
# Generate features for one source
# ============================================================

def generate_features_for_source(
    pairs,
    s1,
    target,
    source,
):

    print()
    print("=" * 60)

    print(
        f"GENERATING FEATURES: S1 -> {source}"
    )

    print("=" * 60)

    source_pairs = pairs[
        pairs["source"] == source
    ].copy()

    if source_pairs.empty:

        print(
            f"No {source} pairs."
        )

        return pd.DataFrame()

    print(
        f"Pairs to process: "
        f"{len(source_pairs):,}"
    )

    results = []

    total = len(source_pairs)

    # --------------------------------------------------------
    # Process in chunks.
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total,
        )

        chunk = source_pairs.iloc[
            start:end
        ].copy()

        print(
            f"  Processing "
            f"{start + 1:,} - {end:,} "
            f"of {total:,}"
        )

        # ----------------------------------------------------
        # Compute pairwise features.
        # ----------------------------------------------------

        feature_chunk = compute_pair_features(
            chunk[
                [
                    "source1_entity_id",
                    "target_entity_id",
                ]
            ],
            s1,
            target,
        )

        # ----------------------------------------------------
        # Add labels.
        # ----------------------------------------------------

        labels = chunk[
            [
                "source1_entity_id",
                "target_entity_id",
                "label",
            ]
        ]

        feature_chunk = feature_chunk.merge(
            labels,
            on=[
                "source1_entity_id",
                "target_entity_id",
            ],
            how="left",
        )

        # ----------------------------------------------------
        # Add source.
        # ----------------------------------------------------

        feature_chunk["source"] = source

        results.append(
            feature_chunk
        )

        del feature_chunk

    if not results:

        return pd.DataFrame()

    result = pd.concat(
        results,
        ignore_index=True,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TRAINING FEATURE GENERATION")
    print("=" * 60)

    # ========================================================
    # Create output directory
    # ========================================================

    TRAINING_CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load candidate pairs
    # ========================================================

    pairs = load_training_pairs()

    # ========================================================
    # Load Source1
    # ========================================================

    s1 = load_required_source1(
        pairs
    )

    # ========================================================
    # Load Source2
    # ========================================================

    s2 = load_required_target_records(
        pairs,
        "S2",
    )

    # ========================================================
    # Load Source3
    # ========================================================

    s3 = load_required_target_records(
        pairs,
        "S3",
    )

    # ========================================================
    # Normalize
    # ========================================================

    (
        s1_norm,
        s2_norm,
        s3_norm,
    ) = normalize_records(
        s1,
        s2,
        s3,
    )

    # ========================================================
    # Generate S2 features
    # ========================================================

    features_s2 = generate_features_for_source(
        pairs,
        s1_norm,
        s2_norm,
        "S2",
    )

    # ========================================================
    # Generate S3 features
    # ========================================================

    features_s3 = generate_features_for_source(
        pairs,
        s1_norm,
        s3_norm,
        "S3",
    )

    # ========================================================
    # Combine
    # ========================================================

    feature_columns = [
        "source1_entity_id",
        "target_entity_id",
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
        "label",
        "source",
    ]

    frames = []

    if not features_s2.empty:

        frames.append(
            features_s2
        )

    if not features_s3.empty:

        frames.append(
            features_s3
        )

    if not frames:

        raise RuntimeError(
            "No features were generated."
        )

    training_features = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Ensure consistent column order.
    # --------------------------------------------------------

    training_features = training_features[
        feature_columns
    ]

    # ========================================================
    # Convert numeric features
    # ========================================================

    numeric_columns = [
        column
        for column in feature_columns
        if column not in [
            "source1_entity_id",
            "target_entity_id",
            "source",
            "label",
        ]
    ]

    for column in numeric_columns:

        training_features[
            column
        ] = pd.to_numeric(
            training_features[column],
            errors="coerce",
        ).fillna(0).astype("float32")

    training_features["label"] = (
        training_features["label"]
        .astype("int8")
    )

    # ========================================================
    # Save
    # ========================================================

    print()
    print("=" * 60)
    print("SAVING TRAINING FEATURES")
    print("=" * 60)

    training_features.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Rows: "
        f"{len(training_features):,}"
    )

    print(
        f"Columns: "
        f"{len(training_features.columns)}"
    )

    print(
        f"Positive pairs: "
        f"{(training_features['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(training_features['label'] == 0).sum():,}"
    )

    print()
    print("Feature columns:")

    for column in training_features.columns:

        print(
            f"  - {column}"
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()