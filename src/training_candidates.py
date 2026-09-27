"""
Memory-safe training candidate generation for the
Amazon Business Entity Resolution Challenge 2026.

Creates a labeled training-pair dataset using:
    - sampled Source1 entities
    - ground-truth positive pairs
    - blocked candidate pairs
    - hard-negative sampling

This is an initial training experiment.
It is not the final test prediction pipeline.
"""

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DIR,
    TRAIN_S1_SAMPLE_SIZE,
    TRAIN_TARGET_SAMPLE_SIZE,
    TARGET_CHUNK_SIZE,
    MAX_NEGATIVES_PER_S1,
    RANDOM_SEED,
    TRAINING_CANDIDATES_DIR,
    TRAINING_PAIRS_FILE,
)

from src.normalization import add_normalized_features
from src.blocking import generate_candidate_pairs


# ============================================================
# Dataset files
# ============================================================

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth():
    """
    Convert ground truth into:

        {
            S1_ID: {S2_ID, S3_ID, ...}
        }
    """

    print("Loading ground truth...")

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if not matched:
            truth[s1_id] = set()

        else:
            truth[s1_id] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

    print(
        f"Ground-truth Source1 entities: "
        f"{len(truth):,}"
    )

    return truth


# ============================================================
# Sample Source1
# ============================================================

def sample_source1(truth):
    """
    Select a reproducible subset of Source1 entities.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    all_ids = np.array(
        list(truth.keys())
    )

    sample_size = min(
        TRAIN_S1_SAMPLE_SIZE,
        len(all_ids),
    )

    selected = rng.choice(
        all_ids,
        size=sample_size,
        replace=False,
    )

    return set(selected)


# ============================================================
# Load selected Source1 rows
# ============================================================

def load_source1_entities(selected_ids):

    print()
    print("Loading selected Source1 records...")

    chunks = []

    for chunk in pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=TARGET_CHUNK_SIZE,
    ):

        selected = chunk[
            chunk["entity_id"].isin(
                selected_ids
            )
        ]

        if not selected.empty:
            chunks.append(selected)

    if not chunks:

        raise RuntimeError(
            "Selected Source1 entities were not found."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(
        f"Loaded Source1 records: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# Collect required positive target IDs
# ============================================================

def collect_required_target_ids(
    selected_ids,
    truth,
    prefix,
):
    """
    Collect all known target IDs required by the selected
    Source1 entities.

    prefix:
        S2 -> Source2
        S3 -> Source3
    """

    required = set()

    for s1_id in selected_ids:

        for target_id in truth.get(
            s1_id,
            set(),
        ):

            if target_id.startswith(prefix):

                required.add(
                    target_id
                )

    return required


# ============================================================
# Load target records
# ============================================================

def load_target_records(
    filepath,
    required_ids,
    random_sample_size,
):
    """
    Read a large target TSV in chunks.

    Always retains required ground-truth target records.

    Also retains additional target records up to the
    configured sample size.
    """

    print()
    print(
        f"Loading target records: "
        f"{filepath.name}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    required_chunks = []
    random_chunks = []

    random_remaining = random_sample_size
    rows_scanned = 0

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=TARGET_CHUNK_SIZE,
    ):

        rows_scanned += len(chunk)

        # ----------------------------------------------------
        # Always retain required positive records
        # ----------------------------------------------------

        if required_ids:

            required = chunk[
                chunk["entity_id"].isin(
                    required_ids
                )
            ]

            if not required.empty:

                required_chunks.append(
                    required
                )

        # ----------------------------------------------------
        # Additional random records
        # ----------------------------------------------------

        if random_remaining > 0:

            take = min(
                random_remaining,
                len(chunk),
            )

            if take > 0:

                indices = rng.choice(
                    len(chunk),
                    size=take,
                    replace=False,
                )

                random_chunks.append(
                    chunk.iloc[indices]
                )

                random_remaining -= take

        if rows_scanned % 1_000_000 == 0:

            print(
                f"  Scanned "
                f"{rows_scanned:,} rows..."
            )

    parts = []

    if required_chunks:

        parts.append(
            pd.concat(
                required_chunks,
                ignore_index=True,
            )
        )

    if random_chunks:

        parts.append(
            pd.concat(
                random_chunks,
                ignore_index=True,
            )
        )

    if not parts:

        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Required and random samples can overlap.
    # --------------------------------------------------------

    result = result.drop_duplicates(
        subset=["entity_id"]
    ).reset_index(drop=True)

    print(
        f"Target records retained: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# Get known positive pairs
# ============================================================

def get_positive_pairs(
    s1,
    target,
    truth,
):
    """
    Return known positive pairs whose target records are
    available in the current target sample.
    """

    target_ids = set(
        target["entity_id"]
    )

    positive_rows = []

    for s1_id in s1["entity_id"]:

        for target_id in truth.get(
            s1_id,
            set(),
        ):

            if target_id in target_ids:

                positive_rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "target_entity_id": target_id,
                    }
                )

    if not positive_rows:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "target_entity_id",
            ]
        )

    return pd.DataFrame(
        positive_rows
    )


# ============================================================
# Prepare candidate pairs
# ============================================================

def prepare_candidates(
    candidates,
    positive_df,
):
    """
    Convert blocking output into the common training schema.

    Blocking output:
        source1_entity_id
        candidate_entity_id

    Training schema:
        source1_entity_id
        target_entity_id
    """

    if candidates.empty:

        return positive_df.copy()

    # --------------------------------------------------------
    # Verify blocking output.
    # --------------------------------------------------------

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in candidates.columns
    ]

    if missing:

        raise RuntimeError(
            "Blocking output is missing columns: "
            + ", ".join(missing)
            + "\nActual columns: "
            + ", ".join(
                candidates.columns
            )
        )

    # --------------------------------------------------------
    # Keep only required columns.
    # --------------------------------------------------------

    candidates = candidates[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Rename blocker ID to training ID.
    # --------------------------------------------------------

    candidates = candidates.rename(
        columns={
            "candidate_entity_id":
                "target_entity_id"
        }
    )

    # --------------------------------------------------------
    # Add known positives explicitly.
    # --------------------------------------------------------

    if not positive_df.empty:

        candidates = pd.concat(
            [
                candidates,
                positive_df,
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Remove duplicate pairs.
    # --------------------------------------------------------

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "target_entity_id",
        ]
    ).reset_index(drop=True)

    return candidates


# ============================================================
# Build labels
# ============================================================

def build_labels(
    candidates,
    truth,
    selected_s1_ids,
):
    """
    Label candidate pairs.

    IMPORTANT:
    Only create the positive lookup for the selected
    Source1 entities.

    This prevents the previous MemoryError caused by
    constructing a lookup for all 2.2M Source1 entities.

    label = 1 -> true match
    label = 0 -> non-match
    """

    if candidates.empty:

        result = candidates.copy()

        result["label"] = pd.Series(
            dtype="int8"
        )

        return result

    # --------------------------------------------------------
    # Small positive lookup.
    #
    # Only 10,000 selected Source1 entities are relevant.
    # --------------------------------------------------------

    positive_lookup = set()

    for s1_id in selected_s1_ids:

        for target_id in truth.get(
            s1_id,
            set(),
        ):

            positive_lookup.add(
                (
                    s1_id,
                    target_id,
                )
            )

    # --------------------------------------------------------
    # Generate labels.
    # --------------------------------------------------------

    labels = []

    for s1_id, target_id in zip(
        candidates["source1_entity_id"],
        candidates["target_entity_id"],
    ):

        labels.append(
            int(
                (
                    s1_id,
                    target_id,
                )
                in positive_lookup
            )
        )

    result = candidates.copy()

    result["label"] = np.asarray(
        labels,
        dtype=np.int8,
    )

    return result


# ============================================================
# Hard-negative sampling
# ============================================================

def keep_hard_negatives(labeled):

    """
    Keep every positive.

    Keep at most MAX_NEGATIVES_PER_S1 negative candidates
    per Source1 entity.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    groups = []

    for _, group in labeled.groupby(
        "source1_entity_id",
        sort=False,
    ):

        positives = group[
            group["label"] == 1
        ]

        negatives = group[
            group["label"] == 0
        ]

        # ----------------------------------------------------
        # Always keep positives.
        # ----------------------------------------------------

        if not positives.empty:

            groups.append(
                positives
            )

        # ----------------------------------------------------
        # Limit negatives.
        # ----------------------------------------------------

        if len(negatives) > MAX_NEGATIVES_PER_S1:

            indices = rng.choice(
                len(negatives),
                size=MAX_NEGATIVES_PER_S1,
                replace=False,
            )

            negatives = negatives.iloc[
                indices
            ]

        if not negatives.empty:

            groups.append(
                negatives
            )

    if not groups:

        return labeled.iloc[
            0:0
        ].copy()

    return pd.concat(
        groups,
        ignore_index=True,
    )


# ============================================================
# Process one target source
# ============================================================

def process_source(
    s1,
    target,
    truth,
    selected_s1_ids,
    source_name,
):

    """
    Generate candidates for:

        S1 -> S2

    or:

        S1 -> S3
    """

    print()
    print("=" * 60)

    print(
        f"PROCESSING S1 -> {source_name}"
    )

    print("=" * 60)

    if target.empty:

        print(
            "No target records available."
        )

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "target_entity_id",
                "label",
                "source",
            ]
        )

    print(
        f"Source1 records : "
        f"{len(s1):,}"
    )

    print(
        f"Target records  : "
        f"{len(target):,}"
    )

    # ========================================================
    # Normalization
    # ========================================================

    print(
        "Normalizing Source1..."
    )

    s1_norm = add_normalized_features(
        s1
    )

    print(
        "Normalizing target..."
    )

    target_norm = add_normalized_features(
        target
    )

    # ========================================================
    # Blocking
    # ========================================================

    print(
        "Generating blocked candidates..."
    )

    candidates = generate_candidate_pairs(
        s1_norm,
        target_norm,
    )

    print(
        f"Blocked candidates: "
        f"{len(candidates):,}"
    )

    # ========================================================
    # Get known positives
    # ========================================================

    positive_df = get_positive_pairs(
        s1,
        target,
        truth,
    )

    print(
        f"Known positives available: "
        f"{len(positive_df):,}"
    )

    # ========================================================
    # Prepare candidates
    # ========================================================

    candidates = prepare_candidates(
        candidates,
        positive_df,
    )

    print(
        f"Unique candidates: "
        f"{len(candidates):,}"
    )

    # ========================================================
    # Labels
    # ========================================================

    labeled = build_labels(
        candidates,
        truth,
        selected_s1_ids,
    )

    print(
        f"Positive candidates: "
        f"{(labeled['label'] == 1).sum():,}"
    )

    print(
        f"Negative candidates: "
        f"{(labeled['label'] == 0).sum():,}"
    )

    # ========================================================
    # Hard-negative sampling
    # ========================================================

    labeled = keep_hard_negatives(
        labeled
    )

    labeled["source"] = source_name

    print(
        f"After hard-negative sampling: "
        f"{len(labeled):,}"
    )

    print(
        f"  Positives: "
        f"{(labeled['label'] == 1).sum():,}"
    )

    print(
        f"  Negatives: "
        f"{(labeled['label'] == 0).sum():,}"
    )

    return labeled


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "MEMORY-SAFE TRAINING CANDIDATE GENERATION"
    )

    print("=" * 60)

    TRAINING_CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Ground truth
    # ========================================================

    truth = load_ground_truth()

    # ========================================================
    # Source1 sample
    # ========================================================

    selected_ids = sample_source1(
        truth
    )

    print()

    print(
        f"Selected Source1 entities: "
        f"{len(selected_ids):,}"
    )

    # ========================================================
    # Source1 records
    # ========================================================

    s1 = load_source1_entities(
        selected_ids
    )

    # ========================================================
    # Required positive target IDs
    # ========================================================

    required_s2 = (
        collect_required_target_ids(
            selected_ids,
            truth,
            "S2",
        )
    )

    required_s3 = (
        collect_required_target_ids(
            selected_ids,
            truth,
            "S3",
        )
    )

    print()

    print(
        f"Required S2 target IDs: "
        f"{len(required_s2):,}"
    )

    print(
        f"Required S3 target IDs: "
        f"{len(required_s3):,}"
    )

    # ========================================================
    # Load Source2
    # ========================================================

    s2 = load_target_records(
        S2_FILE,
        required_s2,
        TRAIN_TARGET_SAMPLE_SIZE,
    )

    # ========================================================
    # Load Source3
    # ========================================================

    s3 = load_target_records(
        S3_FILE,
        required_s3,
        TRAIN_TARGET_SAMPLE_SIZE,
    )

    # ========================================================
    # Process S1 -> S2
    # ========================================================

    pairs_s2 = process_source(
        s1,
        s2,
        truth,
        selected_ids,
        "S2",
    )

    # ========================================================
    # Process S1 -> S3
    # ========================================================

    pairs_s3 = process_source(
        s1,
        s3,
        truth,
        selected_ids,
        "S3",
    )

    # ========================================================
    # Combine
    # ========================================================

    training_pairs = pd.concat(
        [
            pairs_s2,
            pairs_s3,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Save
    # ========================================================

    training_pairs.to_csv(
        TRAINING_PAIRS_FILE,
        sep="\t",
        index=False,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING CANDIDATE GENERATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"Total training pairs: "
        f"{len(training_pairs):,}"
    )

    print(
        f"Positive pairs: "
        f"{(training_pairs['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(training_pairs['label'] == 0).sum():,}"
    )

    print()

    print("Saved:")

    print(
        TRAINING_PAIRS_FILE
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()