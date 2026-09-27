"""
Memory-safe production prediction pipeline.

Amazon ML Challenge 2026
Business Entity Resolution

Uses:
- existing normalization
- existing 7-pass blocking
- trained LightGBM matcher
- optimized threshold = 0.78

Designed for machines with limited RAM.
"""

import gc
import json
import os
import time
from collections import defaultdict

import lightgbm as lgb
import pandas as pd

from src.blocking import generate_candidate_pairs
from src.features import compute_pair_features
from src.normalization import add_normalized_features


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")
MODEL_DIR = "models"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "lightgbm_matcher.txt",
)

THRESHOLD_FILE = os.path.join(
    TRAINING_DIR,
    "threshold_metadata.json",
)

MATCHING_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv",
)

CANDIDATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv",
)

# Conservative sizes for a 15 GB RAM machine.
SOURCE1_CHUNK_SIZE = 5_000
TARGET_CHUNK_SIZE = 50_000
FEATURE_CHUNK_SIZE = 25_000

DEFAULT_THRESHOLD = 0.78

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
# HELPERS
# ============================================================

def count_rows(path):
    """Count TSV data rows without loading the file."""

    with open(path, "rb") as f:
        count = 0

        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            count += chunk.count(b"\n")

    return count


def load_threshold():
    """Load optimized threshold."""

    if not os.path.exists(THRESHOLD_FILE):
        print(
            f"Threshold file not found. "
            f"Using default {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD

    try:
        with open(
            THRESHOLD_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        threshold = float(
            metadata.get(
                "best_threshold",
                DEFAULT_THRESHOLD,
            )
        )

        print(
            f"Loaded optimized threshold: {threshold:.2f}"
        )

        return threshold

    except Exception as exc:
        print(
            f"Could not load threshold: {exc}"
        )
        print(
            f"Using default threshold "
            f"{DEFAULT_THRESHOLD}"
        )

        return DEFAULT_THRESHOLD


def load_model():
    """Load trained LightGBM model."""

    print("Loading LightGBM model...")

    model = lgb.Booster(
        model_file=MODEL_FILE
    )

    print("LightGBM model loaded.")

    return model


def read_chunk(
    path,
    skiprows,
    nrows,
):
    """Read a TSV chunk safely."""

    return pd.read_csv(
        path,
        sep="\t",
        skiprows=range(1, skiprows + 1),
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def write_candidates(
    candidates,
    first_write,
):
    """Append candidates to candidate_pairs.tsv."""

    if candidates.empty:
        return first_write

    output = candidates[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    output = output.rename(
        columns={
            "candidate_entity_id":
                "target_entity_id"
        }
    )

    output = output.drop_duplicates()

    output.to_csv(
        CANDIDATE_OUTPUT,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )

    return False


def predict_candidates(
    candidates,
    df_s1,
    df_target,
    model,
    threshold,
):
    """
    Compute features and predict candidates.

    Returns:
        dict:
            source1_id -> set(target_id)
    """

    if candidates.empty:
        return {}

    predictions = defaultdict(set)

    total = len(candidates)

    for start in range(
        0,
        total,
        FEATURE_CHUNK_SIZE,
    ):

        end = min(
            start + FEATURE_CHUNK_SIZE,
            total,
        )

        batch = candidates.iloc[
            start:end
        ].copy()

        features = compute_pair_features(
            batch,
            df_s1,
            df_target,
        )

        X = features[
            FEATURE_COLUMNS
        ].astype("float32")

        probabilities = model.predict(
            X,
            num_iteration=model.best_iteration,
        )

        features[
            "prediction_probability"
        ] = probabilities

        accepted = features[
            features[
                "prediction_probability"
            ] >= threshold
        ]

        for row in accepted.itertuples(
            index=False
        ):

            predictions[
                row.source1_entity_id
            ].add(
                row.target_entity_id
            )

        del batch
        del features
        del X
        del probabilities
        del accepted

        gc.collect()

    return predictions


# ============================================================
# PROCESS ONE TARGET SOURCE
# ============================================================

def process_target_source(
    source_label,
    target_file,
    s1_ids,
    model,
    threshold,
    matching_dict,
    candidate_first_write,
):
    """
    Process one target source.

    Source1 is processed in small chunks.
    Target source is read in small chunks.
    """

    print()
    print("=" * 70)
    print(
        f"PROCESSING TARGET SOURCE: {source_label}"
    )
    print("=" * 70)

    target_rows = count_rows(
        target_file
    )

    print(
        f"{source_label} rows: "
        f"{target_rows:,}"
    )

    s1_total = len(s1_ids)

    s1_start = 0

    while s1_start < s1_total:

        s1_end = min(
            s1_start + SOURCE1_CHUNK_SIZE,
            s1_total,
        )

        print()
        print(
            f"SOURCE1 CHUNK "
            f"{s1_start // SOURCE1_CHUNK_SIZE + 1}"
            f"/"
            f"{(s1_total + SOURCE1_CHUNK_SIZE - 1) // SOURCE1_CHUNK_SIZE}"
        )

        print(
            f"S1 rows: "
            f"{s1_start:,} - {s1_end - 1:,}"
        )

        # ----------------------------------------------------
        # Load Source1 chunk
        # ----------------------------------------------------

        df_s1 = read_chunk(
            S1_FILE,
            s1_start,
            s1_end - s1_start,
        )

        print(
            "Normalizing Source1 chunk..."
        )

        df_s1 = add_normalized_features(
            df_s1
        )

        print(
            "Source1 chunk normalized."
        )

        # ----------------------------------------------------
        # Process target chunks
        # ----------------------------------------------------

        target_start = 0

        while target_start < target_rows:

            target_end = min(
                target_start + TARGET_CHUNK_SIZE,
                target_rows,
            )

            print(
                f"  {source_label} target "
                f"{target_start:,} - "
                f"{target_end - 1:,}"
            )

            target_chunk = read_chunk(
                target_file,
                target_start,
                target_end - target_start,
            )

            # ------------------------------------------------
            # Normalize target ONCE
            # ------------------------------------------------

            target_chunk = (
                add_normalized_features(
                    target_chunk
                )
            )

            # ------------------------------------------------
            # Blocking
            # ------------------------------------------------

            block_start = time.time()

            candidates = (
                generate_candidate_pairs(
                    df_s1,
                    target_chunk,
                    target_label=source_label,
                )
            )

            block_time = (
                time.time() - block_start
            )

            print(
                f"    Candidates: "
                f"{len(candidates):,} "
                f"({block_time:.2f}s)"
            )

            # ------------------------------------------------
            # Write exact candidate set
            # ------------------------------------------------

            candidate_first_write = (
                write_candidates(
                    candidates,
                    candidate_first_write,
                )
            )

            # ------------------------------------------------
            # ML prediction
            # ------------------------------------------------

            if not candidates.empty:

                prediction_start = time.time()

                accepted = (
                    predict_candidates(
                        candidates,
                        df_s1,
                        target_chunk,
                        model,
                        threshold,
                    )
                )

                prediction_time = (
                    time.time()
                    - prediction_start
                )

                accepted_count = sum(
                    len(v)
                    for v in accepted.values()
                )

                print(
                    f"    Accepted matches: "
                    f"{accepted_count:,} "
                    f"({prediction_time:.2f}s)"
                )

                for (
                    s1_id,
                    target_ids,
                ) in accepted.items():

                    matching_dict[
                        s1_id
                    ].update(
                        target_ids
                    )

            # ------------------------------------------------
            # Free memory
            # ------------------------------------------------

            del target_chunk
            del candidates

            gc.collect()

            target_start = target_end

        # ----------------------------------------------------
        # Free S1 chunk
        # ----------------------------------------------------

        del df_s1

        gc.collect()

        s1_start = s1_end

    return candidate_first_write


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 70)
    print(
        "AMAZON BUSINESS ENTITY RESOLUTION"
    )
    print(
        "MEMORY-SAFE TEST PREDICTION PIPELINE"
    )
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    # --------------------------------------------------------
    # Count Source1
    # --------------------------------------------------------

    print()
    print(
        "Counting test Source1 rows..."
    )

    s1_total = count_rows(
        S1_FILE
    )

    print(
        f"Test Source1 rows: "
        f"{s1_total:,}"
    )

    # --------------------------------------------------------
    # Load Source1 IDs only
    # --------------------------------------------------------

    print()
    print(
        "Loading Source1 entity IDs..."
    )

    s1_ids_df = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=["entity_id"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    s1_ids = (
        s1_ids_df[
            "entity_id"
        ]
        .tolist()
    )

    del s1_ids_df

    gc.collect()

    print(
        f"Stored Source1 IDs: "
        f"{len(s1_ids):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Output initialization
    # --------------------------------------------------------

    if os.path.exists(
        MATCHING_OUTPUT
    ):
        os.remove(
            MATCHING_OUTPUT
        )

    if os.path.exists(
        CANDIDATE_OUTPUT
    ):
        os.remove(
            CANDIDATE_OUTPUT
        )

    print()
    print(
        "Creating candidate_pairs.tsv..."
    )

    # --------------------------------------------------------
    # Matching dictionary
    # --------------------------------------------------------

    matching_dict = defaultdict(set)

    candidate_first_write = True

    # --------------------------------------------------------
    # Source2
    # --------------------------------------------------------

    candidate_first_write = (
        process_target_source(
            "S2",
            S2_FILE,
            s1_ids,
            model,
            threshold,
            matching_dict,
            candidate_first_write,
        )
    )

    # --------------------------------------------------------
    # Source3
    # --------------------------------------------------------

    candidate_first_write = (
        process_target_source(
            "S3",
            S3_FILE,
            s1_ids,
            model,
            threshold,
            matching_dict,
            candidate_first_write,
        )
    )

    # --------------------------------------------------------
    # Write matching results
    # --------------------------------------------------------

    print()
    print(
        "Writing matching_results.tsv..."
    )

    with open(
        MATCHING_OUTPUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "matched_entity_ids\n"
        )

        for s1_id in s1_ids:

            matched = matching_dict.get(
                s1_id,
                set(),
            )

            # Deterministic ordering
            matched_ids = sorted(
                matched
            )

            f.write(
                f"{s1_id}\t"
                f"{','.join(matched_ids)}\n"
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    matched_entities = sum(
        1
        for s1_id in s1_ids
        if matching_dict.get(
            s1_id
        )
    )

    total_matches = sum(
        len(v)
        for v in matching_dict.values()
    )

    singleton_entities = (
        len(s1_ids)
        - matched_entities
    )

    elapsed = (
        time.time()
        - overall_start
    )

    print()
    print("=" * 70)
    print(
        "PREDICTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Source1 entities      : "
        f"{len(s1_ids):,}"
    )

    print(
        f"Entities with matches : "
        f"{matched_entities:,}"
    )

    print(
        f"Singleton predictions : "
        f"{singleton_entities:,}"
    )

    print(
        f"Total predicted links : "
        f"{total_matches:,}"
    )

    print()
    print(
        f"Matching output: "
        f"{MATCHING_OUTPUT}"
    )

    print(
        f"Candidate output: "
        f"{CANDIDATE_OUTPUT}"
    )

    print()
    print(
        f"Total elapsed time: "
        f"{elapsed / 3600:.2f} hours"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()