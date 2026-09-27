"""
Display Missed Ground-Truth Pairs with UTF-8 encoding support on Windows terminal.
"""

import sys
import time
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import generate_candidate_pairs
from src.normalization import add_normalized_features

def display_missed_pairs_sample():
    train_s1_path = PROJECT_ROOT / "dataset/train/train_source1.tsv"
    train_s2_path = PROJECT_ROOT / "dataset/train/train_source2.tsv"
    train_s3_path = PROJECT_ROOT / "dataset/train/train_source3.tsv"
    train_gt_path = PROJECT_ROOT / "dataset/train/train_ground_truth.tsv"

    df_s1 = pd.read_csv(train_s1_path, sep="\t", nrows=1000, dtype=str)
    s1_set = set(df_s1["entity_id"].tolist())
    s1_lookup = df_s1.set_index("entity_id").to_dict("index")

    matching_gt_chunks = []
    for gt_chunk in pd.read_csv(train_gt_path, sep="\t", chunksize=500000, dtype=str):
        sub = gt_chunk[gt_chunk["source1_entity_id"].isin(s1_set)]
        if not sub.empty:
            matching_gt_chunks.append(sub)

    df_gt_correct = pd.concat(matching_gt_chunks, ignore_index=True)

    gt_pairs = set()
    gt_target_ids = set()

    for row in df_gt_correct.itertuples(index=False):
        s1_id = getattr(row, "source1_entity_id")
        matched_str = getattr(row, "matched_entity_ids")
        if pd.isna(matched_str) or not matched_str or not str(matched_str).strip():
            continue
        matched_list = [m.strip() for m in str(matched_str).split(",") if m.strip()]
        for target_id in matched_list:
            gt_pairs.add((s1_id, target_id))
            gt_target_ids.add(target_id)

    # Find Target record metadata for the ground truth targets
    target_lookup = {}
    for target_path in [train_s2_path, train_s3_path]:
        for chunk in pd.read_csv(target_path, sep="\t", chunksize=500000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(gt_target_ids)]
            if not sub.empty:
                for r in sub.to_dict("records"):
                    target_lookup[r["entity_id"]] = r

    df_s1_norm = add_normalized_features(df_s1)

    # Fast re-generation of candidates for df_s1 against the target GT rows
    target_gt_df = pd.DataFrame(list(target_lookup.values()))
    generated_pairs_df = generate_candidate_pairs(df_s1_norm, target_gt_df)
    
    generated_pairs = set()
    for row in generated_pairs_df.itertuples(index=False):
        generated_pairs.add((getattr(row, "source1_entity_id"), getattr(row, "candidate_entity_id")))

    found_matches = gt_pairs & generated_pairs
    missed_matches = gt_pairs - generated_pairs

    print("\n=======================================================")
    print("BLOCKING EVALUATION FINAL SUMMARY")
    print("=======================================================")
    print(f"Total Ground-Truth Matches    : {len(gt_pairs)}")
    print(f"Matches Survived (Found)     : {len(found_matches)}")
    print(f"Missed Ground-Truth Matches  : {len(missed_matches)}")
    print(f"Blocking Recall              : {(len(found_matches)/len(gt_pairs))*100:.2f}% ({len(found_matches)}/{len(gt_pairs)})")

    print("\n-------------------------------------------------------")
    print("SAMPLE OF 10 MISSED GROUND-TRUTH PAIRS")
    print("-------------------------------------------------------")
    for i, (s1_id, tid) in enumerate(list(missed_matches)[:10], 1):
        s1_info = s1_lookup.get(s1_id, {})
        t_info = target_lookup.get(tid, {})

        print(f"\n[Missed Pair {i}]")
        print(f"  S1 ID          : {s1_id}")
        print(f"  S1 Name        : {s1_info.get('business_name', '')}")
        print(f"  S1 Address     : {s1_info.get('business_address', '')}")
        print(f"  S1 Country     : {s1_info.get('country', '')}")
        print(f"  Target ID ({tid[:2]}) : {tid}")
        print(f"  Target Name    : {t_info.get('business_name', '')}")
        print(f"  Target Address : {t_info.get('business_address', '')}")
        print(f"  Target Country : {t_info.get('country', '')}")

if __name__ == "__main__":
    display_missed_pairs_sample()
