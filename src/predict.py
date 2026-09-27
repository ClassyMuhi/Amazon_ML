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

S1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv",
)

S2_FILE = os.path.join(
    TEST_DIR,
    "test_source2.tsv",
)

S3_FILE = os.path.join(
    TEST_DIR,
    "test_source3.tsv",
)

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


# ============================================================
# CHUNK SETTINGS
# ============================================================

SOURCE1_CHUNK_SIZE = 5_000
TARGET_CHUNK_SIZE = 50_000
FEATURE_CHUNK_SIZE = 25_000

DEFAULT_THRESHOLD = 0.78


# ============================================================
# FEATURES USED BY LIGHTGBM
# ============================================================

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
# HELPER: COUNT ROWS
# ============================================================

def count_rows(path):
    """
    Count TSV data rows without loading the complete file.
    """

    with open(
        path,
        "rb",
    ) as f:

        count = 0

        for chunk in iter(
            lambda: f.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            count += chunk.count(
                b"\n"
            )

    return count


# ============================================================
# HELPER: LOAD THRESHOLD
# ============================================================

def load_threshold():
    """
    Load optimized validation threshold.

    Expected:
        output/training/threshold_metadata.json

    Falls back to 0.78 if the file is unavailable.
    """

    if not os.path.exists(
        THRESHOLD_FILE
    ):

        print(
            "Threshold file not found."
        )

        print(
            f"Using default threshold: "
            f"{DEFAULT_THRESHOLD}"
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
            f"Loaded optimized threshold: "
            f"{threshold:.2f}"
        )

        return threshold

    except Exception as exc:

        print(
            f"Could not load threshold: {exc}"
        )

        print(
            f"Using default threshold: "
            f"{DEFAULT_THRESHOLD}"
        )

        return DEFAULT_THRESHOLD


# ============================================================
# HELPER: LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained LightGBM model.
    """

    print(
        "Loading LightGBM model..."
    )

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"LightGBM model not found: "
            f"{MODEL_FILE}"
        )

    model = lgb.Booster(
        model_file=MODEL_FILE
    )

    print(
        "LightGBM model loaded."
    )

    return model


# ============================================================
# HELPER: READ TSV CHUNK
# ============================================================

def read_chunk(
    path,
    skiprows,
    nrows,
):
    """
    Read a TSV chunk safely.
    """

    return pd.read_csv(
        path,
        sep="\t",
        skiprows=range(
            1,
            skiprows + 1,
        ),
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


# ============================================================
# WRITE CANDIDATES
# ============================================================

def write_candidates(
    candidates,
    first_write,
):
    """
    Append the exact generated candidate set.

    Output format:

        source1_entity_id
        target_entity_id
    """

    if candidates.empty:

        return first_write

    # The existing blocker returns:
    #
    # source1_entity_id
    # candidate_entity_id

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
        mode=(
            "w"
            if first_write
            else "a"
        ),
        header=first_write,
    )

    return False


# ============================================================
# PREDICT CANDIDATES
# ============================================================

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

        {
            source1_id: {
                target_id,
                target_id,
                ...
            }
        }
    """

    if candidates.empty:

        return {}

    predictions = defaultdict(
        set
    )

    total = len(
        candidates
    )

    for start in range(
        0,
        total,
        FEATURE_CHUNK_SIZE,
    ):

        end = min(
            start + FEATURE_CHUNK_SIZE,
            total,
        )

        print(
            f"      Feature rows "
            f"{start:,} - {end - 1:,}"
        )

        batch = candidates.iloc[
            start:end
        ].copy()

        # ----------------------------------------------------
        # Compute pair features
        # ----------------------------------------------------

        features = compute_pair_features(
            batch,
            df_s1,
            df_target,
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if features.empty:

            del batch
            del features

            gc.collect()

            continue

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        X = features[
            FEATURE_COLUMNS
        ].astype(
            "float32"
        )

        # ----------------------------------------------------
        # LightGBM prediction
        # ----------------------------------------------------

        probabilities = model.predict(
            X,
            num_iteration=model.best_iteration,
        )

        features[
            "prediction_probability"
        ] = probabilities

        # ----------------------------------------------------
        # Apply optimized threshold
        # ----------------------------------------------------

        accepted = features[
            features[
                "prediction_probability"
            ] >= threshold
        ]

        # ----------------------------------------------------
        # Extract accepted matches
        # ----------------------------------------------------

        for row in accepted.itertuples(
            index=False
        ):

            source1_id = getattr(
                row,
                "source1_entity_id",
                None,
            )

            if source1_id is None:

                continue

            # Depending on the exact version of
            # compute_pair_features(), the target
            # identifier can appear under either
            # name.

            target_id = getattr(
                row,
                "target_entity_id",
                None,
            )

            if target_id is None:

                target_id = getattr(
                    row,
                    "candidate_entity_id",
                    None,
                )

            if target_id is None:

                continue

            predictions[
                source1_id
            ].add(
                target_id
            )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

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

    Source1:
        processed in 5,000-row chunks.

    Target:
        processed in 50,000-row chunks.

    This keeps peak RAM substantially lower than
    loading the complete test data.
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"PROCESSING TARGET SOURCE: "
        f"{source_label}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Count target rows
    # --------------------------------------------------------

    target_rows = count_rows(
        target_file
    )

    print(
        f"{source_label} rows: "
        f"{target_rows:,}"
    )

    # --------------------------------------------------------
    # Source1 information
    # --------------------------------------------------------

    s1_total = len(
        s1_ids
    )

    s1_chunk_number = 0

    s1_start = 0

    total_s1_chunks = (
        s1_total
        + SOURCE1_CHUNK_SIZE
        - 1
    ) // SOURCE1_CHUNK_SIZE

    # --------------------------------------------------------
    # Process Source1 chunks
    # --------------------------------------------------------

    while s1_start < s1_total:

        s1_chunk_number += 1

        s1_end = min(
            s1_start
            + SOURCE1_CHUNK_SIZE,
            s1_total,
        )

        print()
        print(
            f"SOURCE1 CHUNK "
            f"{s1_chunk_number}/"
            f"{total_s1_chunks}"
        )

        print(
            f"S1 rows: "
            f"{s1_start:,} - "
            f"{s1_end - 1:,}"
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

        target_chunk_number = 0

        total_target_chunks = (
            target_rows
            + TARGET_CHUNK_SIZE
            - 1
        ) // TARGET_CHUNK_SIZE

        while target_start < target_rows:

            target_chunk_number += 1

            target_end = min(
                target_start
                + TARGET_CHUNK_SIZE,
                target_rows,
            )

            print(
                f"  {source_label} target "
                f"{target_chunk_number}/"
                f"{total_target_chunks}: "
                f"{target_start:,} - "
                f"{target_end - 1:,}"
            )

            # ------------------------------------------------
            # Load target chunk
            # ------------------------------------------------

            target_chunk = read_chunk(
                target_file,
                target_start,
                target_end - target_start,
            )

            # ------------------------------------------------
            # Normalize target
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
                time.time()
                - block_start
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

                prediction_start = (
                    time.time()
                )

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
                    len(values)
                    for values
                    in accepted.values()
                )

                print(
                    f"    Accepted matches: "
                    f"{accepted_count:,} "
                    f"({prediction_time:.2f}s)"
                )

                # ------------------------------------------------
                # Merge predictions
                # ------------------------------------------------

                for (
                    s1_id,
                    target_ids,
                ) in accepted.items():

                    matching_dict[
                        s1_id
                    ].update(
                        target_ids
                    )

                del accepted

            # ------------------------------------------------
            # Free target memory
            # ------------------------------------------------

            del target_chunk
            del candidates

            gc.collect()

            target_start = target_end

        # ----------------------------------------------------
        # Free Source1 memory
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

    print(
        "=" * 70
    )

    print(
        "AMAZON BUSINESS ENTITY RESOLUTION"
    )

    print(
        "MEMORY-SAFE TEST PREDICTION PIPELINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    # --------------------------------------------------------
    # Count Source1 rows
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
        usecols=[
            "entity_id"
        ],
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
    # Load LightGBM
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Remove incomplete previous outputs
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

    matching_dict = defaultdict(
        set
    )

    candidate_first_write = True

    # ========================================================
    # SOURCE 2
    # ========================================================

    candidate_first_write = (
        process_target_source(
            source_label="S2",
            target_file=S2_FILE,
            s1_ids=s1_ids,
            model=model,
            threshold=threshold,
            matching_dict=matching_dict,
            candidate_first_write=
                candidate_first_write,
        )
    )

    # ========================================================
    # SOURCE 3
    # ========================================================

    candidate_first_write = (
        process_target_source(
            source_label="S3",
            target_file=S3_FILE,
            s1_ids=s1_ids,
            model=model,
            threshold=threshold,
            matching_dict=matching_dict,
            candidate_first_write=
                candidate_first_write,
        )
    )

    # ========================================================
    # WRITE MATCHING RESULTS
    # ========================================================

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

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    matched_entities = sum(
        1
        for s1_id in s1_ids
        if matching_dict.get(
            s1_id
        )
    )

    singleton_entities = (
        len(s1_ids)
        - matched_entities
    )

    total_matches = sum(
        len(values)
        for values
        in matching_dict.values()
    )

    elapsed = (
        time.time()
        - overall_start
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PREDICTION COMPLETE"
    )

    print(
        "=" * 70
    )

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

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()