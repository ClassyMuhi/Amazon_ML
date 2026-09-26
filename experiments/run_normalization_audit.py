"""
Audits normalization on actual training datasets (train_source1, train_source2, train_source3)
using chunked processing for optimal memory efficiency on large-scale datasets.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import (
    normalize_business_name,
    get_core_business_name,
    get_name_tokens,
    normalize_address,
    get_address_tokens,
    extract_numbers,
    extract_postal_code,
    normalize_country,
    add_normalized_features,
)

train_files = [
    ("train_source1", PROJECT_ROOT / "dataset/train/train_source1.tsv"),
    ("train_source2", PROJECT_ROOT / "dataset/train/train_source2.tsv"),
    ("train_source3", PROJECT_ROOT / "dataset/train/train_source3.tsv"),
]

output_dir = PROJECT_ROOT / "output"
output_dir.mkdir(parents=True, exist_ok=True)

audit_results = {}
CHUNK_SIZE = 250000

for label, fpath in train_files:
    print(f"\n=======================================================")
    print(f"Auditing normalization on {label} ({fpath.name})...")
    print(f"=======================================================")
    t0 = time.time()

    total_rows = 0
    names_changed = 0
    core_diff_from_norm = 0
    core_diff_from_raw = 0
    addrs_changed = 0
    postal_extracted = 0
    numbers_extracted = 0
    countries_changed = 0
    examples = []

    # Stream chunks using standard C engine to keep memory low
    chunk_iter = pd.read_csv(fpath, sep="\t", chunksize=CHUNK_SIZE, dtype=str)
    
    first_chunk_saved = False

    for chunk_idx, chunk in enumerate(chunk_iter):
        chunk_rows = len(chunk)
        total_rows += chunk_rows

        # Apply normalization features
        enriched_chunk = add_normalized_features(chunk)

        # Save first 500 rows to output/ as sample demonstration
        if not first_chunk_saved:
            sample_out_path = output_dir / f"sample_normalized_{label}.tsv"
            enriched_chunk.head(500).to_csv(sample_out_path, sep="\t", index=False)
            print(f"  Saved 500-row sample to {sample_out_path.name}")
            first_chunk_saved = True

        raw_names = chunk["business_name"].fillna("").tolist()
        norm_names = enriched_chunk["business_name_normalized"].tolist()
        core_names = enriched_chunk["business_name_core"].tolist()

        raw_addrs = chunk["business_address"].fillna("").tolist()
        norm_addrs = enriched_chunk["business_address_normalized"].tolist()
        postal_codes = enriched_chunk["postal_code"].tolist()
        addr_numbers = enriched_chunk["address_numbers"].tolist()

        raw_countries = chunk["country"].fillna("").tolist()
        norm_countries = enriched_chunk["country_normalized"].tolist()

        # Count transformations
        names_changed += sum(1 for r, n in zip(raw_names, norm_names) if r != n)
        core_diff_from_norm += sum(1 for n, c in zip(norm_names, core_names) if n != c)
        core_diff_from_raw += sum(1 for r, c in zip(raw_names, core_names) if r != c)

        addrs_changed += sum(1 for r, a in zip(raw_addrs, norm_addrs) if r != a)
        postal_extracted += sum(1 for p in postal_codes if p != "")
        numbers_extracted += sum(1 for nums in addr_numbers if len(nums) > 0)
        countries_changed += sum(1 for r, c in zip(raw_countries, norm_countries) if r != c)

        # Collect diverse examples
        if len(examples) < 12:
            for i in range(min(500, chunk_rows)):
                rn, nn, cn = raw_names[i], norm_names[i], core_names[i]
                ra, na, pz = raw_addrs[i], norm_addrs[i], postal_codes[i]
                
                # Check for interesting transformations
                if (nn != cn or any(c in rn for c in ["--", "***", "@", "#", "[[", "&", "+", "É", "é", "À", "à", "ம", "म"])) and len(examples) < 12:
                    examples.append({
                        "entity_id": str(chunk.iloc[i]["entity_id"]),
                        "original_name": rn,
                        "normalized_name": nn,
                        "core_name": cn,
                        "original_address": ra,
                        "normalized_address": na,
                        "extracted_numbers": addr_numbers[i],
                        "postal_code": pz,
                        "country": norm_countries[i]
                    })

        if (chunk_idx + 1) % 4 == 0 or total_rows >= 2000000:
            print(f"  Processed {total_rows:,} rows... ({time.time() - t0:.1f}s)")

    duration = time.time() - t0
    print(f"Completed {label}: {total_rows:,} rows in {duration:.2f}s ({total_rows / duration:,.0f} rows/s)")

    audit_results[label] = {
        "total_rows": total_rows,
        "processing_time_sec": round(duration, 2),
        "throughput_rows_per_sec": round(total_rows / duration, 0),
        "transformations": {
            "business_names_changed": {
                "count": names_changed,
                "percentage": round(names_changed / total_rows * 100, 2)
            },
            "legal_suffixes_removed": {
                "count": core_diff_from_norm,
                "percentage": round(core_diff_from_norm / total_rows * 100, 2)
            },
            "total_core_name_changes": {
                "count": core_diff_from_raw,
                "percentage": round(core_diff_from_raw / total_rows * 100, 2)
            },
            "addresses_changed": {
                "count": addrs_changed,
                "percentage": round(addrs_changed / total_rows * 100, 2)
            },
            "postal_codes_extracted": {
                "count": postal_extracted,
                "percentage": round(postal_extracted / total_rows * 100, 2)
            },
            "address_numbers_extracted": {
                "count": numbers_extracted,
                "percentage": round(numbers_extracted / total_rows * 100, 2)
            },
            "countries_changed": {
                "count": countries_changed,
                "percentage": round(countries_changed / total_rows * 100, 2)
            }
        },
        "examples": examples
    }

audit_summary_path = output_dir / "normalization_audit_summary.json"
with open(audit_summary_path, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"NORMALIZATION AUDIT COMPLETE. Summary saved to {audit_summary_path.name}")
print("=" * 80)
for label, res in audit_results.items():
    t = res["transformations"]
    print(f"\n{label} ({res['total_rows']:,} rows in {res['processing_time_sec']}s):")
    print(f"  - Business names transformed: {t['business_names_changed']['count']:,} ({t['business_names_changed']['percentage']}%)")
    print(f"  - Legal suffixes stripped:   {t['legal_suffixes_removed']['count']:,} ({t['legal_suffixes_removed']['percentage']}%)")
    print(f"  - Addresses transformed:      {t['addresses_changed']['count']:,} ({t['addresses_changed']['percentage']}%)")
    print(f"  - Postal codes identified:   {t['postal_codes_extracted']['count']:,} ({t['postal_codes_extracted']['percentage']}%)")
    print(f"  - Address numbers found:     {t['address_numbers_extracted']['count']:,} ({t['address_numbers_extracted']['percentage']}%)")
