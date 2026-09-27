"""
Correct Ground-Truth Recall Evaluation Script for src/blocking.py
using the 1,000-row Source 1 sample from train_source1.tsv.
"""

import sys
import time
import re
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "PROJECT_ROOT" not in globals() else PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import generate_candidate_pairs
from src.normalization import (
    add_normalized_features,
    normalize_business_name,
    get_core_business_name,
    normalize_address,
    extract_numbers,
    extract_postal_code,
    normalize_country,
)


def evaluate_blocking_recall():
    print("=======================================================", flush=True)
    print("Evaluating Blocking Recall on Ground Truth (1,000 S1 Sample)...", flush=True)
    print("=======================================================", flush=True)

    train_s1_path = PROJECT_ROOT / "dataset/train/train_source1.tsv"
    train_s2_path = PROJECT_ROOT / "dataset/train/train_source2.tsv"
    train_s3_path = PROJECT_ROOT / "dataset/train/train_source3.tsv"
    train_gt_path = PROJECT_ROOT / "dataset/train/train_ground_truth.tsv"

    if not train_s1_path.exists() or not train_gt_path.exists():
        print("Error: Training dataset files not found!", flush=True)
        return

    # 1. Load 1,000 S1 records
    t0 = time.time()
    print("Loading 1,000 S1 records...", flush=True)
    df_s1 = pd.read_csv(train_s1_path, sep="\t", nrows=1000, dtype=str)
    s1_set = set(df_s1["entity_id"].tolist())
    s1_lookup = df_s1.set_index("entity_id").to_dict("index")

    # 2. Load Ground Truth corresponding to those exact 1,000 S1 records
    print("Loading Ground Truth matching those 1,000 S1 records...", flush=True)
    matching_gt_chunks = []
    for gt_chunk in pd.read_csv(train_gt_path, sep="\t", chunksize=500000, dtype=str):
        sub = gt_chunk[gt_chunk["source1_entity_id"].isin(s1_set)]
        if not sub.empty:
            matching_gt_chunks.append(sub)

    df_gt_correct = pd.concat(matching_gt_chunks, ignore_index=True)

    gt_pairs = set()
    gt_s2_ids = set()
    gt_s3_ids = set()

    for row in df_gt_correct.itertuples(index=False):
        s1_id = getattr(row, "source1_entity_id")
        matched_str = getattr(row, "matched_entity_ids")
        if pd.isna(matched_str) or not matched_str or not str(matched_str).strip():
            continue
        matched_list = [m.strip() for m in str(matched_str).split(",") if m.strip()]
        for target_id in matched_list:
            gt_pairs.add((s1_id, target_id))
            if target_id.startswith("S2-"):
                gt_s2_ids.add(target_id)
            elif target_id.startswith("S3-"):
                gt_s3_ids.add(target_id)

    total_gt_matches = len(gt_pairs)
    print(f"Correct Ground-Truth matches loaded for 1,000 S1 records: {total_gt_matches}", flush=True)
    print(f"  - S2 ground-truth matches: {len(gt_s2_ids)}", flush=True)
    print(f"  - S3 ground-truth matches: {len(gt_s3_ids)}", flush=True)

    # Pre-normalize df_s1
    df_s1_norm = add_normalized_features(df_s1)
    s1_raw_countries = set(df_s1["country"].fillna("").astype(str).str.strip().unique())

    # Build first token of core name set for S1 records (min len >= 2)
    s1_core_first_tokens = set()
    for core in df_s1_norm["business_name_core"]:
        toks = core.split()
        if toks and len(toks[0]) >= 2:
            s1_core_first_tokens.add(toks[0])

    s1_exact_cores = set(c for c in df_s1_norm["business_name_core"] if c and len(c) >= 3)
    s1_postals = set(p for p in df_s1_norm["postal_code"] if p)

    print(f"S1 Search Keys: {len(s1_core_first_tokens)} core first-tokens, {len(s1_exact_cores)} exact cores, {len(s1_postals)} postals", flush=True)

    def is_potential_candidate(name_str: str, addr_str: str) -> bool:
        if not name_str or not isinstance(name_str, str):
            return False
        words = set(re.findall(r"[a-z0-9]+", name_str.lower()))
        if words & s1_core_first_tokens:
            return True
        if s1_postals:
            addr_words = set(re.findall(r"[a-z0-9]+", str(addr_str).lower()))
            if addr_words & s1_postals:
                return True
        return False

    target_sources = [("S2", train_s2_path, gt_s2_ids), ("S3", train_s3_path, gt_s3_ids)]
    all_generated_pairs = set()
    target_lookup = {}

    CHUNK_SIZE = 500000

    for label, target_path, target_gt_ids in target_sources:
        print(f"\nProcessing {label} dataset ({target_path.name})...", flush=True)
        t_src = time.time()
        chunk_candidates_count = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(target_path, sep="\t", chunksize=CHUNK_SIZE, dtype=str)):
            t_ch = time.time()
            # Store target records that are in GT for detailed missed pair output
            gt_rows = chunk[chunk["entity_id"].isin(target_gt_ids)]
            if not gt_rows.empty:
                for row_dict in gt_rows.to_dict("records"):
                    target_lookup[row_dict["entity_id"]] = row_dict

            # Fast Country & Token Filter
            c_mask = chunk["country"].fillna("").astype(str).str.strip().isin(s1_raw_countries)
            gt_mask = chunk["entity_id"].isin(target_gt_ids)

            bnames = chunk["business_name"].tolist()
            baddrs = chunk["business_address"].tolist()
            cand_mask = [is_potential_candidate(bn, ba) for bn, ba in zip(bnames, baddrs)]

            filtered_chunk = chunk[c_mask & (pd.Series(cand_mask, index=chunk.index) | gt_mask)]

            if not filtered_chunk.empty:
                c_pairs = generate_candidate_pairs(df_s1_norm, filtered_chunk)
                if not c_pairs.empty:
                    for p in c_pairs.itertuples(index=False):
                        all_generated_pairs.add((getattr(p, "source1_entity_id"), getattr(p, "candidate_entity_id")))
                    chunk_candidates_count += len(c_pairs)

            print(f"  Chunk {chunk_idx+1} processed in {time.time() - t_ch:.2f}s (Filtered {len(filtered_chunk)} rows, Cumulative candidates: {len(all_generated_pairs):,})", flush=True)

        print(f"Finished {label}: generated {chunk_candidates_count:,} candidates in {time.time() - t_src:.2f}s", flush=True)

    total_candidates = len(all_generated_pairs)
    avg_candidates_per_s1 = total_candidates / len(df_s1)

    # 3. Compute Metrics
    found_matches = gt_pairs & all_generated_pairs
    missed_matches = gt_pairs - all_generated_pairs

    found_count = len(found_matches)
    missed_count = len(missed_matches)
    blocking_recall = (found_count / total_gt_matches) if total_gt_matches > 0 else 0.0

    # 4. Print Summary Report
    print("\n=======================================================", flush=True)
    print("BLOCKING EVALUATION SUMMARY (1,000 S1 RECORDS)", flush=True)
    print("=======================================================", flush=True)
    print(f"1. Total ground-truth matches         : {total_gt_matches}", flush=True)
    print(f"2. Ground-truth matches found        : {found_count}", flush=True)
    print(f"3. Missed ground-truth matches        : {missed_count}", flush=True)
    print(f"4. Blocking recall                    : {blocking_recall * 100:.2f}% ({found_count}/{total_gt_matches})", flush=True)
    print(f"5. Total candidate pairs generated    : {total_candidates:,}", flush=True)
    print(f"6. Average candidates per S1 entity   : {avg_candidates_per_s1:.2f}", flush=True)

    # 5. Display 10 Missed Ground-Truth Pairs if any exist
    if missed_count > 0:
        print("\n-------------------------------------------------------", flush=True)
        print("SAMPLE OF 10 MISSED GROUND-TRUTH PAIRS", flush=True)
        print("-------------------------------------------------------", flush=True)
        for i, (s1_id, tid) in enumerate(list(missed_matches)[:10], 1):
            s1_info = s1_lookup.get(s1_id, {})
            t_info = target_lookup.get(tid, {})

            print(f"\n[Missed Pair {i}]", flush=True)
            print(f"  S1 ID          : {s1_id}", flush=True)
            print(f"  S1 Name        : {s1_info.get('business_name', '')}", flush=True)
            print(f"  S1 Address     : {s1_info.get('business_address', '')}", flush=True)
            print(f"  S1 Country     : {s1_info.get('country', '')}", flush=True)
            print(f"  Target ID ({tid[:2]}) : {tid}", flush=True)
            print(f"  Target Name    : {t_info.get('business_name', '')}", flush=True)
            print(f"  Target Address : {t_info.get('business_address', '')}", flush=True)
            print(f"  Target Country : {t_info.get('country', '')}", flush=True)


if __name__ == "__main__":
    evaluate_blocking_recall()
