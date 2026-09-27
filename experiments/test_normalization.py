"""
Comprehensive test script for src/normalization.py.

Validates all 10 normalization functions across standard examples,
user-requested edge cases, missing data, numbers, punctuation,
multilingual unicode characters, and DataFrame enrichment.
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import (
    normalize_text,
    normalize_business_name,
    get_name_tokens,
    get_core_business_name,
    normalize_address,
    get_address_tokens,
    extract_numbers,
    extract_postal_code,
    normalize_country,
    add_normalized_features,
)


def run_tests():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE NORMALIZATION UNIT TESTS")
    print("=" * 80)

    # 1. User Specified Examples
    print("\n[TEST 1] User-Specified Name Normalization Examples:")
    ex1 = "ABC PVT. LTD."
    ex2 = "ABC Pvt Ltd"
    ex3 = "  ABC   CORPORATION  "
    ex4 = "12, M.G. Road, Chennai - 600001"

    print(f"  Raw: {ex1!r}")
    print(f"    -> Normalized: {normalize_business_name(ex1)!r}")
    print(f"    -> Core:       {get_core_business_name(ex1)!r}")
    print(f"    -> Tokens:     {get_name_tokens(normalize_business_name(ex1))!r}")

    print(f"  Raw: {ex2!r}")
    print(f"    -> Normalized: {normalize_business_name(ex2)!r}")
    print(f"    -> Core:       {get_core_business_name(ex2)!r}")
    print(f"    -> Tokens:     {get_name_tokens(normalize_business_name(ex2))!r}")

    print(f"  Raw: {ex3!r}")
    print(f"    -> Normalized: {normalize_business_name(ex3)!r}")
    print(f"    -> Core:       {get_core_business_name(ex3)!r}")
    print(f"    -> Tokens:     {get_name_tokens(normalize_business_name(ex3))!r}")

    # Verify that all 3 variants resolve to the exact same core name "abc"
    core1 = get_core_business_name(ex1)
    core2 = get_core_business_name(ex2)
    core3 = get_core_business_name(ex3)
    assert core1 == core2 == core3 == "abc", f"Failed: {core1} != {core2} != {core3}"
    print("  [PASS] All 3 variants resolved to identical core name: 'abc'")

    print(f"\n[TEST 2] User-Specified Address Example: {ex4!r}")
    norm_addr4 = normalize_address(ex4)
    tokens_addr4 = get_address_tokens(norm_addr4)
    nums_addr4 = extract_numbers(ex4)
    zip_addr4 = extract_postal_code(ex4)

    print(f"    -> Normalized:  {norm_addr4!r}")
    print(f"    -> Tokens:      {tokens_addr4!r}")
    print(f"    -> Numbers:     {nums_addr4!r}")
    print(f"    -> Postal Code: {zip_addr4!r}")

    assert "12" in nums_addr4 and "600001" in nums_addr4, "Failed number extraction"
    assert zip_addr4 == "600001", f"Failed postal code extraction: {zip_addr4}"
    print("  [PASS] Correctly extracted numbers ['12', '600001'] and PIN '600001'")

    # 2B. Postal Code Extraction & False Positive Rejection
    print("\n[TEST 2B] Postal Code Extraction & Street Number False-Positive Rejection:")
    postal_false_positives = [
        # User specified test cases
        ("17160 Presbyterian Road", ""),
        ("10500 Bergtold Road", ""),
        ("23037 Olympia Drive", ""),
        ("34233 River Rd", ""),
        # Dataset sample cases
        ("WI, Town Of Townsend, 17160 Presbyterian Road", ""),
        ("Unit Suite 235, 10500 Bergtold Road, NY, Clarence", ""),
        ("VA, 23037 Olympia Drive, Loudoun County", ""),
        ("Long Neck, 34233 River Rd, Delaware", ""),
        ("445243 995 Road, Gore, OK", ""),
        ("Unit STE A-07, Bedford County, VA, 18013 Forest Road", ""),
        ("Grove, NY, 11852 16", ""),
        ("OK, Spiro, 24062 Tucker Road", ""),
        ("Katy, TX, 21342 Bending Green Way", ""),
        ("OH, 12313 REXFORD AVE, CLEVELAND", ""),
        ("NO 00925 APARTMENT D-26, BANGALORE, Karnataka", ""),
        ("PRINCE WILLIAM COUNTY, VA, 11261 LADY JANE LOIP", ""),
        ("AR, HETH, 00148 ST FRANCIS 519 ROAD", ""),
        ("OK, 18987 4185 Rd, CLAREMORE", ""),
        ("NY, 01080 87, CATSKILL", ""),
        ("NULL, 18611 EACH ELM WAY, TX, HOSTON", ""),
        ("H.N. 04162, SECTOR D, PKT, VASANT KUNJ, SOUTH DELHI, Delhi", ""),
        ("NEW DELHI, Delhi, 00311", ""),
        ("BIG LAKE, 25412 167 1/2 ST, MN", ""),
        ("NOBLESVILLE, 15769 EXPLORATION BLVD, IN", ""),
        ("Edwall, Washington, 29511 Tucker Prairie Rd", ""),
        ("Phoenix, AZ, 16007 45th Street", ""),
        ("Washington, Kent, 18223 286th Ct", ""),
        ("Texas, 14659 Coffee Ln, Scurry", ""),
    ]
    for raw_a, exp_code in postal_false_positives:
        code = extract_postal_code(raw_a)
        assert code == exp_code, f"False positive detected: {raw_a!r} -> got {code!r}, expected {exp_code!r}"
    print("  [PASS] All street/building numbers correctly rejected (postal_code == '')")

    # Genuine postal codes
    genuine_postal_cases = [
        ("12, M.G. Road, Chennai - 600001", "600001"),
        ("33466 Warwick Hills Road, Yucaipa, CA 92399", "92399"),
        ("3315 FREMONT ST, PEORIA, IL 61602", "61602"),
        ("PO Box 90210, Beverly Hills, CA 90210-1234", "90210-1234"),
        ("10 Downing St, London SW1A 2AA, UK", "SW1A 2AA"),
        ("1425 Rue Chabanel O, Montréal, QC H4N 2S7", "H4N2S7"),
        ("24 Rue de la Paix, 75008 Paris", "75008"),
        ("Near Bus Stand, PIN Code: 560001, Bangalore", "560001"),
        ("Austin, TX 78701, USA", "78701"),
        ("Plot 42, Sector 18, Gurgaon - 122001", "122001"),
    ]
    for raw_a, exp_code in genuine_postal_cases:
        code = extract_postal_code(raw_a)
        assert code == exp_code, f"Genuine postal code failed: {raw_a!r} -> got {code!r}, expected {exp_code!r}"
    print("  [PASS] Genuine postal codes correctly extracted (US ZIP, ZIP+4, PIN, UK, Canada, France)")

    # 3. Missing Values (NaN, None, empty string, whitespace)
    print("\n[TEST 3] Missing / Null Value Handling:")
    for null_val in [None, float("nan"), "", "   "]:
        assert normalize_text(null_val) == "", f"Failed on {null_val!r}"
        assert normalize_business_name(null_val) == "", f"Failed on {null_val!r}"
        assert get_name_tokens(normalize_business_name(null_val)) == [], f"Failed on {null_val!r}"
        assert get_core_business_name(null_val) == "", f"Failed on {null_val!r}"
        assert normalize_address(null_val) == "", f"Failed on {null_val!r}"
        assert get_address_tokens(normalize_address(null_val)) == [], f"Failed on {null_val!r}"
        assert extract_numbers(null_val) == [], f"Failed on {null_val!r}"
        assert extract_postal_code(null_val) == "", f"Failed on {null_val!r}"
        assert normalize_country(null_val) == "", f"Failed on {null_val!r}"
    print("  [PASS] All null/empty/NaN representations safely returned empty strings/lists without errors")

    # 3. Numbers in Names
    print("\n[TEST 4] Numbers in Names (Distinction Preservation):")
    n_examples = ["3M Company", "7-Eleven Store Inc.", "Studio 54 LLC", "#4 Presidio"]
    for ne in n_examples:
        norm = normalize_business_name(ne)
        core = get_core_business_name(ne)
        tokens = get_name_tokens(norm)
        print(f"  Raw: {ne!r} -> Norm: {norm!r} | Core: {core!r} | Tokens: {tokens!r}")
        assert any(any(c.isdigit() for c in t) for t in tokens), f"Numbers lost in {ne}"
    print("  [PASS] Numbers preserved in all business names")

    # 4. Numbers in Addresses
    print("\n[TEST 5] Complex Numbers in Addresses:")
    addr_examples = [
        ("6(29), C.I.T. Colony, 2Nd Main Road", ["6", "29", "2nd"]),
        ("18/479E/3 Peringattil Tower Kangarappady", ["18/479e/3"]),
        ("Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203", ["af-684", "989", "9487203"]),
        ("33466 Warwick Hills Road, Yucaipa, CA 92399", ["33466", "92399"]),
        ("IA, Iowa City, 1064 Newton Rd, Unit 11", ["1064", "11"]),
    ]
    for raw_addr, expected_subset in addr_examples:
        extracted = extract_numbers(raw_addr)
        print(f"  Address: {raw_addr!r} -> Extracted Numbers: {extracted!r}")
        for exp in expected_subset:
            assert exp in extracted, f"Expected {exp} in {extracted} for {raw_addr}"
    print("  [PASS] House, plot, unit, and phone numbers accurately extracted")

    # 6. Noise & Punctuation
    print("\n[TEST 6] Noise, Symbols, and Punctuation:")
    punct_cases = [
        ("-- Holloway Peak Seafood Inc", "holloway peak seafood inc", "holloway peak seafood"),
        ("*** Country Real LLC", "country real llc", "country real"),
        ("Obsidian, [[LLC]]", "obsidian llc", "obsidian"),
        ("[(Service)] Womens Health Group, LLC", "service womens health group llc", "service womens health group"),
        ("Chordia & Partners", "chordia and partners", "chordia and partners"),
        ("PAYNE-ENRTPRMISES", "payne enrtprmises", "payne enrtprmises"),
        ("Apex Summit / Peak", "apex summit peak", "apex summit peak"),
    ]
    for raw_p, exp_norm, exp_core in punct_cases:
        norm = normalize_business_name(raw_p)
        core = get_core_business_name(raw_p)
        print(f"  {raw_p!r} -> Norm: {norm!r} | Core: {core!r}")
        assert norm == exp_norm, f"Norm mismatch: {norm!r} != {exp_norm!r}"
        assert core == exp_core, f"Core mismatch: {core!r} != {exp_core!r}"
    print("  [PASS] Noise prefixes, bracketed tags, hyphens, and ampersands handled cleanly")

    # 6. Unicode & Multilingual Scripts
    print("\n[TEST 7] Unicode & Multilingual Scripts:")
    unicode_cases = [
        ("Payne Énterprises", "payne enterprises", "payne enterprises"),  # Accent stripped
        ("Dréxkor LLC", "drexkor llc", "drexkor"),                        # Accent stripped + suffix stripped
        ("Fractales Amis Groupe S.A.S", "fractales amis groupe sas", "fractales amis groupe"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग"), # Hindi script + Hindi suffix
        ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்"), # Tamil script + Tamil suffix
    ]
    for raw_u, exp_norm, exp_core in unicode_cases:
        norm = normalize_business_name(raw_u)
        core = get_core_business_name(raw_u)
        print(f"  {raw_u} -> Norm: {norm} | Core: {core}")
        assert norm == exp_norm, f"Unicode norm mismatch: {norm!r} != {exp_norm!r}"
        assert core == exp_core, f"Unicode core mismatch: {core!r} != {exp_core!r}"
    print("  [PASS] Unicode accents decomposed to base ASCII; native Indic scripts preserved and legal suffixes stripped")

    # 7. Country Normalization (Open-Set)
    print("\n[TEST 8] Open-Set Country Normalization:")
    country_tests = [
        ("US", "US"),
        ("us", "US"),
        ("  uS  ", "US"),
        ("India", "India"),
        ("INDIA", "India"),
        ("france", "France"),
        ("FRANCE", "France"),
        ("FR", "FR"),
        ("germany", "Germany"),
        ("DE", "DE"),
        ("", ""),
        (None, ""),
    ]
    for raw_c, exp_c in country_tests:
        norm_c = normalize_country(raw_c)
        assert norm_c == exp_c, f"Country norm failed: {norm_c!r} != {exp_c!r}"
    print("  [PASS] Open-set country normalization works uniformly across codes and names")

    # 8. DataFrame Processing
    print("\n[TEST 9] add_normalized_features DataFrame Function:")
    sample_df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "ABC PVT. LTD.",
            "business_address": "12, M.G. Road, Chennai - 600001",
            "country": "India"
        },
        {
            "entity_id": "S1-002",
            "business_name": "Payne Énterprises",
            "business_address": "3315 FREMONT ST, PEORIA, IL 61602",
            "country": "US"
        },
        {
            "entity_id": "S2-003",
            "business_name": "  ABC   CORPORATION  ",
            "business_address": None,
            "country": "india"
        },
        {
            "entity_id": "S3-004",
            "business_name": "Dréxkor LLC",
            "business_address": "85 Wayne Ave, Ticonderoga, NY",
            "country": "US"
        }
    ])

    enriched_df = add_normalized_features(sample_df)

    # Verify original columns intact
    for col in ["entity_id", "business_name", "business_address", "country"]:
        assert col in enriched_df.columns, f"Original column {col} missing"
        assert enriched_df[col].tolist() == sample_df[col].tolist(), f"Original column {col} modified!"

    # Verify new columns present
    expected_new_cols = [
        "business_name_normalized",
        "business_name_core",
        "business_name_tokens",
        "business_address_normalized",
        "address_tokens",
        "address_numbers",
        "postal_code",
        "country_normalized"
    ]
    for ncol in expected_new_cols:
        assert ncol in enriched_df.columns, f"Expected derived column {ncol} not found"

    print("\nEnriched DataFrame sample output:")
    for idx, row in enriched_df.iterrows():
        print(f"\nRecord {row['entity_id']}:")
        print(f"  Orig Name:    {row['business_name']}")
        print(f"  Norm Name:    {row['business_name_normalized']}")
        print(f"  Core Name:    {row['business_name_core']}")
        print(f"  Name Tokens:  {row['business_name_tokens']}")
        print(f"  Orig Addr:    {row['business_address']}")
        print(f"  Norm Addr:    {row['business_address_normalized']}")
        print(f"  Addr Tokens:  {row['address_tokens']}")
        print(f"  Addr Numbers: {row['address_numbers']}")
        print(f"  Postal Code:  {row['postal_code']}")
        print(f"  Norm Country: {row['country_normalized']}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
