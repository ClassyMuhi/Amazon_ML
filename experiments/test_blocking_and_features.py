"""
Verification test for src/blocking.py and src/features.py.

Runs candidate generation and pairwise feature computation on a sample from dataset/train
as well as synthetic cross-country edge cases (including France, US, India).
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import add_normalized_features
from src.blocking import generate_candidate_pairs
from src.features import compute_pair_features


def test_blocking_and_features_synthetic():
    print("=======================================================")
    print("Running Synthetic Test (US, India, France)...")
    print("=======================================================")

    s1_data = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Acme Industrial Solutions Inc.",
            "business_address": "123 Main St, Springfield, IL 62701",
            "country": "US",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Rajesh Enterprises Pvt Ltd",
            "business_address": "45 M.G. Road, Bangalore - 560001",
            "country": "India",
        },
        {
            "entity_id": "S1-3",
            "business_name": "Lumière Boulangerie SARL",
            "business_address": "12 Rue de la Paix, 75002 Paris",
            "country": "France",
        },
    ])

    s2_data = pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name": "Acme Industrial Solutions LLC",
            "business_address": "123 Main Street, Springfield, IL 62701",
            "country": "US",
        },
        {
            "entity_id": "S2-102",
            "business_name": "Rajesh Enterprises",
            "business_address": "45 MG Road, Bengaluru 560001",
            "country": "India",
        },
        {
            "entity_id": "S2-103",
            "business_name": "Lumiere Boulangerie",
            "business_address": "12 Rue de la Paix, Paris 75002",
            "country": "France",
        },
        {
            "entity_id": "S2-104",
            "business_name": "Random Unrelated Company",
            "business_address": "999 Other St, London",
            "country": "UK",
        },
    ])

    print("\n--- Generating Candidate Pairs ---")
    candidates = generate_candidate_pairs(s1_data, s2_data, target_label="S2")
    print(candidates)

    assert not candidates.empty, "Candidate generation returned empty DataFrame!"
    assert len(candidates) == 3, f"Expected 3 candidate pairs, got {len(candidates)}"
    print("Blocking test passed: 3 true pairs found across US, India, France without cross-country noise.")

    print("\n--- Computing Features ---")
    features = compute_pair_features(candidates, s1_data, s2_data)
    print(features.to_string())

    assert len(features) == 3, f"Expected 3 feature rows, got {len(features)}"
    assert "name_levenshtein_ratio" in features.columns
    assert "postal_code_match" in features.columns
    assert "is_address_missing" in features.columns
    assert (features["country_match"] == 1.0).all(), "Country match should be 1.0 for all candidates"
    assert (features["postal_code_match"] == 1.0).all(), "Postal codes should match for test samples"
    print("Features test passed!")


def test_blocking_and_features_real_data():
    print("\n=======================================================")
    print("Running Real Dataset Sample Test...")
    print("=======================================================")

    train_s1_path = PROJECT_ROOT / "dataset/train/train_source1.tsv"
    train_s2_path = PROJECT_ROOT / "dataset/train/train_source2.tsv"

    if not train_s1_path.exists() or not train_s2_path.exists():
        print("Real train dataset files not found, skipping real sample test.")
        return

    print("Loading 1,000 rows sample from train_source1 and train_source2...")
    df_s1 = pd.read_csv(train_s1_path, sep="\t", nrows=1000, dtype=str)
    df_s2 = pd.read_csv(train_s2_path, sep="\t", nrows=5000, dtype=str)

    print("Generating candidate pairs...")
    candidates = generate_candidate_pairs(df_s1, df_s2)
    print(f"Generated {len(candidates)} candidate pairs for 1,000 S1 records.")

    if not candidates.empty:
        print("Computing features for sample candidate pairs...")
        features = compute_pair_features(candidates.head(50), df_s1, df_s2)
        print(f"Computed {len(features.columns)} features for sample pair subset:")
        print(features.head(5).to_string())

    print("\nReal Dataset Sample Test completed successfully!")


if __name__ == "__main__":
    test_blocking_and_features_synthetic()
    test_blocking_and_features_real_data()
