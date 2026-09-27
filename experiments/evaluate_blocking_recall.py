"""
Memory-safe blocking recall evaluation.

Evaluates the existing blocking logic on a representative subset of
the training dataset against the corresponding ground-truth matches.

IMPORTANT:
This is a diagnostic experiment, not the final training pipeline.
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import add_normalized_features
from src.blocking import generate_candidate_pairs


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

S1_SAMPLE_SIZE = 10_000
TARGET_SAMPLE_SIZE = 100_000
GT_CHUNK_SIZE = 250_000

S1_PATH = PROJECT_ROOT / "dataset/train/train_source1.tsv"
S2_PATH = PROJECT_ROOT / "dataset/train/train_source2.tsv"
S3_PATH = PROJECT_ROOT / "dataset/train/train_source3.tsv"
GT_PATH = PROJECT_ROOT / "dataset/train/train_ground_truth.tsv"

OUTPUT_DIR = PROJECT_ROOT / "output"

MISSED_PATH = OUTPUT_DIR / "blocking_missed_pairs_sample.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "blocking_candidates_sample.tsv"


# ---------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------

def load_ground_truth_for_s1(s1_ids):
    """
    Load ground-truth pairs only for the sampled S1 entities.
    """

    s1_ids = set(s1_ids)

    gt_pairs_s2 = set()
    gt_pairs_s3 = set()

    print("\nLoading ground truth...")

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=GT_CHUNK_SIZE,
    ):
        chunk = chunk[
            chunk["source1_entity_id"].isin(s1_ids)
        ]

        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id
            matched = row.matched_entity_ids

            if pd.isna(matched):
                continue

            matched = str(matched).strip()

            if not matched:
                continue

            for target_id in matched.split(","):

                target_id = target_id.strip()

                if not target_id:
                    continue

                if target_id.startswith("S2-"):
                    gt_pairs_s2.add(
                        (s1_id, target_id)
                    )

                elif target_id.startswith("S3-"):
                    gt_pairs_s3.add(
                        (s1_id, target_id)
                    )

    print(
        f"Ground-truth S1→S2 pairs: {len(gt_pairs_s2):,}"
    )

    print(
        f"Ground-truth S1→S3 pairs: {len(gt_pairs_s3):,}"
    )

    return gt_pairs_s2, gt_pairs_s3


# ---------------------------------------------------------
# Load target records required by ground truth
# ---------------------------------------------------------

def load_target_records(path, target_ids):
    """
    Read target TSV in chunks and keep only records whose IDs
    occur in the sampled ground truth.

    This avoids loading the complete 5M-row target dataset.
    """

    target_ids = set(target_ids)

    pieces = []

    print(f"\nFinding {len(target_ids):,} target records in:")
    print(path.name)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=GT_CHUNK_SIZE,
    ):

        sub = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        if not sub.empty:
            pieces.append(sub)

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    print(
        f"Loaded {len(result):,} required target records."
    )

    return result


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_source(
    source_name,
    s1_df,
    target_df,
    ground_truth,
):

    print("\n" + "=" * 60)
    print(f"EVALUATING S1 → {source_name}")
    print("=" * 60)

    if not ground_truth:
        print("No ground-truth matches found.")
        return set(), set()

    start = time.time()

    print("Normalizing S1 sample...")

    s1_norm = add_normalized_features(s1_df)

    print("Normalizing target records...")

    target_norm = add_normalized_features(target_df)

    print("Generating candidates...")

    candidates = generate_candidate_pairs(
        s1_norm,
        target_norm,
        target_label=source_name,
    )

    elapsed = time.time() - start

    generated = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    found = ground_truth & generated
    missed = ground_truth - generated

    recall = (
        len(found) / len(ground_truth)
        if ground_truth
        else 1.0
    )

    print("\nResults")
    print("-" * 60)

    print(
        f"Ground-truth matches : {len(ground_truth):,}"
    )

    print(
        f"Candidate pairs      : {len(generated):,}"
    )

    print(
        f"Recovered matches    : {len(found):,}"
    )

    print(
        f"Missed matches       : {len(missed):,}"
    )

    print(
        f"Blocking recall      : {recall * 100:.2f}%"
    )

    print(
        f"Candidates / S1     : "
        f"{len(generated) / len(s1_df):.2f}"
    )

    print(
        f"Runtime              : {elapsed:.2f} sec"
    )

    return generated, missed


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("MEMORY-SAFE BLOCKING RECALL EVALUATION")
    print("=" * 60)

    print(f"S1 sample size     : {S1_SAMPLE_SIZE:,}")
    print(f"Target sample size : {TARGET_SAMPLE_SIZE:,}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load S1 sample
    # -----------------------------------------------------

    print("\nLoading S1 sample...")

    s1_df = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        nrows=S1_SAMPLE_SIZE,
    )

    print(
        f"Loaded {len(s1_df):,} S1 records."
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    gt_s2, gt_s3 = load_ground_truth_for_s1(
        s1_df["entity_id"].tolist()
    )

    all_s2_ids = {
        target_id
        for _, target_id in gt_s2
    }

    all_s3_ids = {
        target_id
        for _, target_id in gt_s3
    }

    # -----------------------------------------------------
    # Load only target records required for evaluation
    # -----------------------------------------------------

    s2_df = load_target_records(
        S2_PATH,
        all_s2_ids,
    )

    s3_df = load_target_records(
        S3_PATH,
        all_s3_ids,
    )

    # -----------------------------------------------------
    # Evaluate S2
    # -----------------------------------------------------

    candidates_s2, missed_s2 = evaluate_source(
        "S2",
        s1_df,
        s2_df,
        gt_s2,
    )

    # -----------------------------------------------------
    # Evaluate S3
    # -----------------------------------------------------

    candidates_s3, missed_s3 = evaluate_source(
        "S3",
        s1_df,
        s3_df,
        gt_s3,
    )

    # -----------------------------------------------------
    # Save candidates
    # -----------------------------------------------------

    all_candidates = candidates_s2 | candidates_s3

    if all_candidates:

        candidate_df = pd.DataFrame(
            list(all_candidates),
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
        )

        candidate_df["target_source"] = (
            candidate_df["candidate_entity_id"]
            .str[:2]
        )

        candidate_df.to_csv(
            CANDIDATE_PATH,
            sep="\t",
            index=False,
        )

    # -----------------------------------------------------
    # Save misses
    # -----------------------------------------------------

    all_missed = missed_s2 | missed_s3

    if all_missed:

        missed_df = pd.DataFrame(
            list(all_missed),
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
        )

        missed_df["target_source"] = (
            missed_df["candidate_entity_id"]
            .str[:2]
        )

        missed_df.to_csv(
            MISSED_PATH,
            sep="\t",
            index=False,
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    total_gt = len(gt_s2) + len(gt_s3)
    total_missed = len(all_missed)
    total_found = total_gt - total_missed

    overall_recall = (
        total_found / total_gt
        if total_gt
        else 1.0
    )

    print("\n" + "=" * 60)
    print("FINAL BLOCKING RECALL SUMMARY")
    print("=" * 60)

    print(
        f"S1 records evaluated : {len(s1_df):,}"
    )

    print(
        f"S2 ground-truth      : {len(gt_s2):,}"
    )

    print(
        f"S2 missed             : {len(missed_s2):,}"
    )

    print(
        f"S3 ground-truth      : {len(gt_s3):,}"
    )

    print(
        f"S3 missed             : {len(missed_s3):,}"
    )

    print(
        f"Total ground-truth   : {total_gt:,}"
    )

    print(
        f"Total recovered      : {total_found:,}"
    )

    print(
        f"Total missed         : {total_missed:,}"
    )

    print(
        f"Overall recall       : "
        f"{overall_recall * 100:.2f}%"
    )

    print("\nOutput files:")

    if CANDIDATE_PATH.exists():
        print(f"  {CANDIDATE_PATH}")

    if MISSED_PATH.exists():
        print(f"  {MISSED_PATH}")


if __name__ == "__main__":
    main()