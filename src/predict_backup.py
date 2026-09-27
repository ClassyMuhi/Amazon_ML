"""
Memory-safe test prediction pipeline for Amazon Business Entity Resolution.

Processes Source1 in chunks so that the full 1.73M-row Source1 dataset
does not have to be normalized in memory simultaneously.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate_candidate_pairs
from .features import compute_pair_features
from .normalization import add_normalized_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

MODEL_FILE = MODEL_DIR / "lightgbm_matcher.txt"
THRESHOLD_FILE = OUTPUT_DIR / "training" / "threshold_results.tsv"

MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# MEMORY SETTINGS
# ============================================================

# Your machine has ~4.3 GB free RAM.
# Keep these conservative.
SOURCE1_CHUNK_SIZE = 5_000
TARGET_CHUNK_SIZE = 50_000


# ============================================================
# MODEL FEATURES
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
# THRESHOLD
# ============================================================

DEFAULT_THRESHOLD = 0.78


def load_threshold() -> float:
    """
    Load optimized threshold from threshold_results.tsv.
    """

    if not THRESHOLD_FILE.exists():
        print(
            f"Threshold file not found: {THRESHOLD_FILE}"
        )
        print(
            f"Using default threshold: {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD

    try:
        df = pd.read_csv(
            THRESHOLD_FILE,
            sep="\t"
        )

        if "macro_f05" not in df.columns:
            print(
                "macro_f05 column not found."
            )
            print(
                f"Using default threshold: {DEFAULT_THRESHOLD}"
            )
            return DEFAULT_THRESHOLD

        best_row = df.loc[
            df["macro_f05"].idxmax()
        ]

        threshold = float(
            best_row["threshold"]
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
            f"Using default threshold: {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD


# ============================================================
# LOAD SOURCE1 CHUNKS
# ============================================================

def get_source1_columns() -> list[str]:
    """
    Determine Source1 columns from the header.
    """

    header = pd.read_csv(
        S1_FILE,
        sep="\t",
        nrows=0
    )

    return list(header.columns)


def count_source1_rows() -> int:
    """
    Count Source1 rows without loading the full dataset.
    """

    count = 0

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        # Skip header
        next(f, None)

        for _ in f:
            count += 1

    return count


# ============================================================
# LOAD TARGET CHUNKS
# ============================================================

def iter_target_chunks(
    file_path: Path,
    chunk_size: int,
):
    """
    Yield normalized target chunks.

    Only one target chunk is kept in memory at a time.
    """

    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    for chunk_number, df in enumerate(
        reader,
        start=1
    ):

        print(
            f"      Loading target chunk {chunk_number} "
            f"({len(df):,} rows)"
        )

        df = add_normalized_features(df)

        yield df

        del df
        gc.collect()


# ============================================================
# PROCESS ONE S1 CHUNK AGAINST ONE TARGET SOURCE
# ============================================================

def process_s1_chunk_against_source(
    df_s1_chunk: pd.DataFrame,
    target_file: Path,
    target_label: str,
    model: lgb.Booster,
    threshold: float,
    candidate_file_handle,
    accepted_matches: dict[str, set[str]],
):
    """
    Process one Source1 chunk against all chunks of one target source.
    """

    source1_chunk_number = (
        df_s1_chunk["entity_id"]
        .astype(str)
        .iloc[0]
        if len(df_s1_chunk) > 0
        else "EMPTY"
    )

    print(
        f"\n    Processing {target_label} "
        f"for S1 chunk starting at {source1_chunk_number}"
    )

    target_chunk_count = 0
    total_candidates = 0
    total_matches = 0

    for df_target in iter_target_chunks(
        target_file,
        TARGET_CHUNK_SIZE,
    ):

        target_chunk_count += 1

        # ----------------------------------------------------
        # BLOCKING
        # ----------------------------------------------------

        candidates = generate_candidate_pairs(
            df_s1_chunk,
            df_target,
            target_label=target_label,
        )

        if candidates is None or len(candidates) == 0:

            del df_target
            gc.collect()

            continue

        # ----------------------------------------------------
        # STANDARDIZE CANDIDATE ID
        # ----------------------------------------------------

        if (
            "candidate_entity_id" in candidates.columns
            and "target_entity_id"
            not in candidates.columns
        ):

            candidates = candidates.rename(
                columns={
                    "candidate_entity_id":
                    "target_entity_id"
                }
            )

        if "target_source" not in candidates.columns:

            candidates["target_source"] = target_label

        # ----------------------------------------------------
        # WRITE FINAL CANDIDATE SET
        # ----------------------------------------------------

        candidate_output = candidates[
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        ].copy()

        candidate_output.to_csv(
            candidate_file_handle,
            sep="\t",
            index=False,
            header=False,
        )

        total_candidates += len(
            candidate_output
        )

        # ----------------------------------------------------
        # FEATURE GENERATION
        # ----------------------------------------------------

        features_df = compute_pair_features(
            candidates,
            df_s1_chunk,
            df_target,
        )

        if len(features_df) == 0:

            del candidates
            del candidate_output
            del features_df
            del df_target

            gc.collect()

            continue

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        X = features_df[
            FEATURE_COLUMNS
        ].astype(
            np.float32,
            copy=False,
        )

        probabilities = model.predict(
            X
        )

        # ----------------------------------------------------
        # APPLY THRESHOLD
        # ----------------------------------------------------

        accepted_mask = (
            probabilities >= threshold
        )

        if accepted_mask.any():

            accepted_df = features_df.loc[
                accepted_mask,
                [
                    "source1_entity_id",
                    "target_entity_id",
                ],
            ]

            for row in accepted_df.itertuples(
                index=False
            ):

                s1_id = str(
                    row.source1_entity_id
                )

                target_id = str(
                    row.target_entity_id
                )

                accepted_matches[
                    s1_id
                ].add(target_id)

                total_matches += 1

            del accepted_df

        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del X
        del probabilities
        del accepted_mask
        del features_df
        del candidates
        del candidate_output
        del df_target

        gc.collect()

    print(
        f"    {target_label}: "
        f"{total_candidates:,} candidates, "
        f"{total_matches:,} accepted matches "
        f"across {target_chunk_count} target chunks"
    )


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

def write_matching_results(
    all_source1_ids: list[str],
    accepted_matches: dict[str, set[str]],
):
    """
    Write exactly one row for every Source1 entity.
    """

    print(
        "\nWriting matching_results.tsv..."
    )

    rows = []

    for s1_id in all_source1_ids:

        matches = accepted_matches.get(
            s1_id,
            set()
        )

        if matches:

            # Deterministic ordering
            matched_entity_ids = ",".join(
                sorted(matches)
            )

        else:

            matched_entity_ids = ""

        rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids":
                    matched_entity_ids,
            }
        )

        # Flush periodically so that this list
        # doesn't become unnecessarily large.
        if len(rows) >= 100_000:

            df = pd.DataFrame(rows)

            df.to_csv(
                MATCHING_FILE,
                sep="\t",
                index=False,
                mode="a",
                header=not MATCHING_FILE.exists(),
            )

            rows.clear()

            del df

            gc.collect()

    if rows:

        df = pd.DataFrame(rows)

        df.to_csv(
            MATCHING_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=not MATCHING_FILE.exists(),
        )

        del df

        rows.clear()

        gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AMAZON BUSINESS ENTITY RESOLUTION"
    )
    print(
        "MEMORY-SAFE TEST PREDICTION PIPELINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = load_threshold()

    print(
        f"\nPrediction threshold: {threshold:.2f}"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        S1_FILE,
        S2_FILE,
        S3_FILE,
        MODEL_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # REMOVE OLD OUTPUTS
    # --------------------------------------------------------

    if MATCHING_FILE.exists():

        print(
            f"\nRemoving old {MATCHING_FILE}"
        )

        MATCHING_FILE.unlink()

    if CANDIDATE_FILE.exists():

        print(
            f"Removing old {CANDIDATE_FILE}"
        )

        CANDIDATE_FILE.unlink()

    # --------------------------------------------------------
    # COUNT SOURCE1
    # --------------------------------------------------------

    print(
        "\nCounting test Source1 rows..."
    )

    total_s1_rows = count_source1_rows()

    print(
        f"Test Source1 rows: {total_s1_rows:,}"
    )

    # --------------------------------------------------------
    # LOAD SOURCE1 IDs ONLY
    # --------------------------------------------------------

    print(
        "\nLoading Source1 entity IDs..."
    )

    s1_ids_df = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=["entity_id"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    all_source1_ids = (
        s1_ids_df["entity_id"]
        .astype(str)
        .tolist()
    )

    del s1_ids_df

    gc.collect()

    print(
        f"Stored Source1 IDs: "
        f"{len(all_source1_ids):,}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\nLoading LightGBM model..."
    )

    model = lgb.Booster(
        model_file=str(
            MODEL_FILE
        )
    )

    print(
        "LightGBM model loaded."
    )

    # --------------------------------------------------------
    # ACCEPTED MATCHES
    # --------------------------------------------------------

    accepted_matches = defaultdict(set)

    # --------------------------------------------------------
    # PREPARE CANDIDATE FILE
    # --------------------------------------------------------

    print(
        "\nCreating candidate_pairs.tsv..."
    )

    candidate_file_handle = open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline="",
    )

    candidate_file_handle.write(
        "source1_entity_id\t"
        "target_entity_id\n"
    )

    # --------------------------------------------------------
    # READ SOURCE1 IN CHUNKS
    # --------------------------------------------------------

    print(
        "\nStarting chunked prediction..."
    )

    s1_reader = pd.read_csv(
        S1_FILE,
        sep="\t",
        chunksize=SOURCE1_CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    total_chunks = (
        total_s1_rows +
        SOURCE1_CHUNK_SIZE -
        1
    ) // SOURCE1_CHUNK_SIZE

    for chunk_number, df_s1_chunk in enumerate(
        s1_reader,
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"SOURCE1 CHUNK "
            f"{chunk_number}/{total_chunks}"
        )

        print(
            f"Rows: {len(df_s1_chunk):,}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # NORMALIZE ONLY THIS S1 CHUNK
        # ----------------------------------------------------

        print(
            "Normalizing Source1 chunk..."
        )

        df_s1_chunk = add_normalized_features(
            df_s1_chunk
        )

        print(
            "Source1 chunk normalized."
        )

        # ----------------------------------------------------
        # S1 → S2
        # ----------------------------------------------------

        process_s1_chunk_against_source(
            df_s1_chunk=df_s1_chunk,
            target_file=S2_FILE,
            target_label="S2",
            model=model,
            threshold=threshold,
            candidate_file_handle=
                candidate_file_handle,
            accepted_matches=accepted_matches,
        )

        # ----------------------------------------------------
        # S1 → S3
        # ----------------------------------------------------

        process_s1_chunk_against_source(
            df_s1_chunk=df_s1_chunk,
            target_file=S3_FILE,
            target_label="S3",
            model=model,
            threshold=threshold,
            candidate_file_handle=
                candidate_file_handle,
            accepted_matches=accepted_matches,
        )

        # ----------------------------------------------------
        # RELEASE CHUNK
        # ----------------------------------------------------

        del df_s1_chunk

        gc.collect()

        print(
            f"\nCompleted Source1 chunk "
            f"{chunk_number}/{total_chunks}"
        )

        print(
            f"Accepted matches accumulated: "
            f"{sum(len(v) for v in accepted_matches.values()):,}"
        )

    # --------------------------------------------------------
    # CLOSE CANDIDATE FILE
    # --------------------------------------------------------

    candidate_file_handle.close()

    # --------------------------------------------------------
    # WRITE FINAL MATCHING RESULTS
    # --------------------------------------------------------

    write_matching_results(
        all_source1_ids,
        accepted_matches,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_matches = sum(
        len(matches)
        for matches in accepted_matches.values()
    )

    matched_s1 = sum(
        1
        for matches in accepted_matches.values()
        if matches
    )

    singleton_s1 = (
        len(all_source1_ids)
        - matched_s1
    )

    print("\n" + "=" * 70)
    print(
        "PREDICTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Source1 entities       : "
        f"{len(all_source1_ids):,}"
    )

    print(
        f"S1 entities with match : "
        f"{matched_s1:,}"
    )

    print(
        f"S1 singleton entities  : "
        f"{singleton_s1:,}"
    )

    print(
        f"Total accepted matches : "
        f"{total_matches:,}"
    )

    print(
        f"\nCandidate file:"
        f"\n  {CANDIDATE_FILE}"
    )

    print(
        f"\nMatching file:"
        f"\n  {MATCHING_FILE}"
    )

    print(
        "\nThreshold used: "
        f"{threshold:.2f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()