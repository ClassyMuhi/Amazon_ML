# Dataset Analysis: Amazon Business Entity Resolution Challenge 2026

> [!NOTE]
> This document presents a comprehensive, empirical exploratory data analysis (EDA) across all train and test datasets for the Amazon Business Entity Resolution Challenge 2026. The findings are based entirely on strict data analysis without any prior modeling, synthetic data generation, or external lookups.

---

## 1. Dataset Overview

The competition dataset consists of business records collected across multiple disparate sources (**Source 1**, **Source 2**, and **Source 3**). The goal is to perform entity resolution: linking each **Source 1** entity to its true corresponding records in **Source 2** and **Source 3**.

### File Verification & Actual File Paths
During the initial file verification phase, all paths were checked:
- **Expected `dataset/test/source3.tsv` vs Actual:** The file referenced as `dataset/test/source3.tsv` is actually named **`dataset/test/test_source3.tsv`** on disk. All other 6 files match their expected standard naming.
- All files are UTF-8 encoded text files using tab (`\t`) as the delimiter.

### Summary Table of All Files
| Split | File Name | Actual File Path | File Size | Line Count | Total Rows (Excl. Header) | Columns |
|:---|:---|:---|---:|---:|---:|:---:|
| **Train** | `train_source1.tsv` | `dataset/train/train_source1.tsv` | 200.34 MB | 2,206,822 | **2,206,821** | 4 |
| **Train** | `train_source2.tsv` | `dataset/train/train_source2.tsv` | 466.63 MB | 5,034,617 | **5,034,616** | 4 |
| **Train** | `train_source3.tsv` | `dataset/train/train_source3.tsv` | 480.37 MB | 5,285,604 | **5,285,603** | 4 |
| **Train** | `train_ground_truth.tsv` | `dataset/train/train_ground_truth.tsv` | 121.13 MB | 2,206,822 | **2,206,821** | 2 |
| **Test** | `test_source1.tsv` | `dataset/test/test_source1.tsv` | 166.91 MB | 1,732,545 | **1,732,544** | 4 |
| **Test** | `test_source2.tsv` | `dataset/test/test_source2.tsv` | 485.86 MB | 4,887,274 | **4,887,273** | 4 |
| **Test** | `test_source3.tsv` | `dataset/test/test_source3.tsv` | 482.56 MB | 5,082,317 | **5,082,316** | 4 |
| **Total** | **7 Files** | — | **2.35 GB** | **26,435,999** | **26,435,994** | — |

---

## 2. File Statistics

A detailed per-file audit was executed over all 26.4 million records across the 6 source files and the ground truth file.

### Comprehensive Source Files Statistics Table
| Metric | `train_source1` | `train_source2` | `train_source3` | `test_source1` | `test_source2` | `test_source3` |
|:---|---:|---:|---:|---:|---:|---:|
| **Total Rows** | 2,206,821 | 5,034,616 | 5,285,603 | 1,732,544 | 4,887,273 | 5,082,316 |
| **Total Columns** | 4 | 4 | 4 | 4 | 4 | 4 |
| **Unique Entity IDs** | 2,206,821 | 5,034,616 | 5,285,603 | 1,732,544 | 4,887,273 | 5,082,316 |
| **Duplicate Entity IDs** | **0** | **0** | **0** | **0** | **0** | **0** |
| **Exact Duplicate Rows** | **0** | **0** | **0** | **0** | **0** | **0** |
| **Unique Business Names** | 1,539,229 | 4,402,009 | 4,651,609 | 1,238,867 | 4,311,041 | 4,521,929 |
| **Unique Business Addresses** | 2,130,606 | 4,337,261 | 4,632,764 | 1,677,483 | 4,224,783 | 4,456,435 |
| **Missing Business Names** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Missing Business Addresses** | 0 (0.0%) | **168,967 (3.36%)** | **175,916 (3.33%)** | 0 (0.0%) | **129,408 (2.65%)** | **136,098 (2.68%)** |
| **Missing Country** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Name Len (Min / Max / Avg)** | 3 / 105 / 24.03 | 2 / 104 / 25.10 | 2 / 123 / 25.20 | 3 / 92 / 23.84 | 2 / 102 / 25.70 | 2 / 103 / 25.66 |
| **Addr Len (Min / Max / Avg)** | 11 / 256 / 52.07 | 0 / 249 / 46.23 | 0 / 240 / 46.71 | 11 / 268 / 57.21 | 0 / 269 / 50.41 | 0 / 267 / 48.74 |
| **Non-Empty Addr Avg Len** | 52.07 | 47.83 | 48.32 | 57.21 | 51.78 | 50.08 |

---

## 3. Schema Analysis

### Source Files Schema (`train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3`)
All source files share the exact same 4-column relational schema:

| Column Name | Inferred Type | Python/Storage Type | Description | Constraints & Nullability |
|:---|:---|:---|:---|:---|
| `entity_id` | String / Identifier | UTF-8 String | Unique entity identifier prefixed with source token (`S1-`, `S2-`, `S3-`) followed by integer ID | Primary Key; **100% Unique; No Nulls** |
| `business_name` | String / Text | UTF-8 String | Commercial/trading name, legal name, or online DBA | **100% Populated; No Nulls**; multilingual scripts present |
| `business_address` | String / Text | UTF-8 String | Street, locality, city, state, postal code, or regional address | Fully populated in Source 1; **2.6% - 3.4% Missing in Source 2 and Source 3** |
| `country` | Categorical / Open-set | UTF-8 String | Country identifier (`US`, `India`, `France`, etc.) | **100% Populated; No Nulls**; Open-set |

### Ground Truth Schema (`train_ground_truth.tsv`)
| Column Name | Inferred Type | Python/Storage Type | Description | Constraints & Nullability |
|:---|:---|:---|:---|:---|
| `source1_entity_id` | String / Identifier | UTF-8 String | The reference Source 1 entity identifier (`S1-<number>`) | Primary Key; Exactly 2,206,821 unique rows (1:1 with `train_source1.tsv`) |
| `matched_entity_ids` | String / Delimited List | UTF-8 String | Comma-separated list of true corresponding entity IDs in Source 2 and/or Source 3 | Empty string for non-matches (5.58%); otherwise list of `S2-*` and `S3-*` IDs |

---

## 4. Missing Values Analysis

| Source File | Total Records | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
|:---|---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 2,206,821 | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| `train_source2.tsv` | 5,034,616 | 0 (0.00%) | 0 (0.00%) | **168,967 (3.36%)** | 0 (0.00%) |
| `train_source3.tsv` | 5,285,603 | 0 (0.00%) | 0 (0.00%) | **175,916 (3.33%)** | 0 (0.00%) |
| `test_source1.tsv` | 1,732,544 | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| `test_source2.tsv` | 4,887,273 | 0 (0.00%) | 0 (0.00%) | **129,408 (2.65%)** | 0 (0.00%) |
| `test_source3.tsv` | 5,082,316 | 0 (0.00%) | 0 (0.00%) | **136,098 (2.68%)** | 0 (0.00%) |

### Key Missing Data Insights:
1. **Source 1 is completely clean and complete:** Neither train nor test `source1` contains a single missing name, address, or country.
2. **Missing Addresses in S2 & S3:** Across train and test, **~2.6% to 3.4% of all records in Source 2 and Source 3 have an empty string address (`""`)**. 
3. **No Missing Names or Countries:** Across all 24.2M records across all 6 files, `business_name` and `country` are **100% present**.

---

## 5. Duplicate Analysis

### Row & Entity ID Duplication
- **Duplicate Rows:** **0 duplicate rows** exist in any file. Every row is distinct.
- **Duplicate Entity IDs:** **0 duplicate entity IDs** exist within any source file. Every entity ID appears exactly once in its respective file.

### Repeated Business Names in Source 1
While IDs are unique, business names frequently repeat:
- **Total records in `train_source1`:** 2,206,821
- **Distinct business names:** 1,539,229
- **Names appearing exactly once:** 1,361,436 (88.45% of distinct names)
- **Names appearing multiple times:** 177,793 (11.55% of distinct names)
- **Records sharing a name with at least one other record:** **845,385 records (38.31% of Source 1!)**

#### Top 20 Most Frequent Names in `train_source1`:
| Rank | Business Name | Occurrences in Source 1 |
|:---|:---|:---:|
| 1 | Primary Care Group | 253 |
| 2 | Ear Nose & Throat Group | 251 |
| 3 | Pediatric Group | 222 |
| 4 | Womens Health Group | 220 |
| 5 | Physical Therapy Group | 218 |
| 6 | Pediatric Dental Group | 216 |
| 7 | Behavioral Health Group | 215 |
| 8 | Chiropractic Group | 209 |
| 9 | Orthopedic Group | 208 |
| 10 | Eye Group | 207 |
| 11 | Dental Group | 205 |
| 12 | Family Group | 204 |
| 13 | Dermatology Group | 202 |
| 14 | Vision Group | 199 |
| 15 | Pediatric Dentistry Group | 198 |
| 16 | Urgent Care Group | 198 |
| 17 | Internal Medicine Group | 197 |
| 18 | Foot & Ankle Group | 191 |
| 19 | Oncology Group | 188 |
| 20 | Urology Group | 185 |

> [!WARNING]
> Nearly **38.3% of Source 1 businesses share a name with another entity** (primarily generic professional practices, healthcare clinics, retail franchises, and corporate structures). Matching solely based on name similarity without strict address and geographic gating will lead to immense false positive rates!

---

## 6. Country Analysis (Open-Set Evaluation)

> [!IMPORTANT]
> The evaluation confirms the user's warning: **the country field is strictly OPEN-SET**. While the training set contains only `US` and `India`, the test set introduces new countries such as **`France`**. Algorithms must never hard-code geographic partitions.

### Distribution Across All Datasets

| Dataset | Total Rows | US (%) | India (%) | France (%) | Notes |
|:---|---:|---:|---:|---:|:---|
| `train_source1.tsv` | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | Train: 2 countries |
| `train_source2.tsv` | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | Train: 2 countries |
| `train_source3.tsv` | 5,285,603 | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) | Train: 2 countries |
| `test_source1.tsv` | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | **259,452 (14.98%)** | **Test: France introduced!** |
| `test_source2.tsv` | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | **703,378 (14.39%)** | **Test: France introduced!** |
| `test_source3.tsv` | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | **731,615 (14.40%)** | **Test: France introduced!** |

### Observations:
- **Zero Cross-Country Matches:** In ground truth analysis, entities strictly match within the same country partition.
- **Distribution Shift in Test:** In the training split, US accounts for ~60% and India ~40%. In the test split, India accounts for ~47.3%, US accounts for ~38.3%, and France accounts for ~14.4%.

---

## 7. Ground Truth Analysis (`train_ground_truth.tsv`)

### Core Metrics
- **Total Ground Truth Rows:** **2,206,821** (Matches `train_source1.tsv` row-for-row).
- **Unique Source 1 Entities in Ground Truth:** **2,206,821** (100% coverage, 0 duplicates).
- **Format Validation:**
  - Unexpected Source 1 ID formats: **0** (All match `^S1-\d+$`).
  - Unexpected matched ID formats: **0** (All match `^(S2|S3)-\d+$`).
  - In-row duplicate IDs: **0** (No repeated IDs in any comma-separated list).
  - Cross-row ID collisions: **0** (No S2 or S3 entity is mapped to more than one S1 entity).

### Cardinality Breakdown
| Category | Count | Percentage of Source 1 |
|:---|---:|---:|
| **Zero Matches (No corresponding S2 or S3 record)** | **123,247** | **5.58%** |
| **Exactly 1 Match** | **119,157** | **5.40%** |
| **Multiple Matches (2 or more matched records)** | **1,964,417** | **89.02%** |

### Detailed Match Count Distribution (Number of S2 + S3 Matches per S1 Entity)
| Match Count | Number of S1 Entities | Percentage | Cumulative % |
|:---|---:|---:|---:|
| **0 matches** | 123,247 | 5.58% | 5.58% |
| **1 match** | 119,157 | 5.40% | 10.98% |
| **2 matches** | 375,212 | 17.00% | 27.99% |
| **3 matches** | 530,841 | 24.05% | 52.04% |
| **4 matches** | 484,115 | 21.94% | 73.98% |
| **5 matches** | 321,957 | 14.59% | 88.57% |
| **6 matches** | 164,868 | 7.47% | 96.04% |
| **7 matches** | 63,968 | 2.90% | 98.94% |
| **8 matches** | 18,680 | 0.85% | 99.78% |
| **9 matches** | 4,205 | 0.19% | 99.97% |
| **10 matches** | 534 | 0.02% | 100.00% |
| **11 matches** | 37 | 0.00% | 100.00% |
| **Max Matches** | **11 matches** | — | — |

### Source-Level Matched Entity Breakdown
| Metric | Count | Source Total | Match Coverage Rate |
|:---|---:|---:|---:|
| **Matched S2 Entities (Unique)** | **3,693,619** | 5,034,616 | **73.36%** of Source 2 |
| **Unmatched S2 Entities (Distractors/Noise)** | **1,340,997** | 5,034,616 | **26.64%** of Source 2 |
| **Matched S3 Entities (Unique)** | **3,944,746** | 5,285,603 | **74.63%** of Source 3 |
| **Unmatched S3 Entities (Distractors/Noise)** | **1,340,857** | 5,285,603 | **25.37%** of Source 3 |

### Source Overlap Patterns Among Source 1 Entities
- **Matched to S2 ONLY:** 143,029 entities (6.48%)
- **Matched to S3 ONLY:** 164,498 entities (7.45%)
- **Matched to BOTH S2 and S3:** **1,776,047 entities (80.48%)**
- **Matched to NEITHER (0 matches):** 123,247 entities (5.58%)

---

## 8. Business Name Analysis

### Length Statistics
| Dataset | Min Len | Max Len | Mean Len | Median Len | Casing Distribution |
|:---|---:|---:|---:|---:|:---|
| `train_source1` | 3 | 105 | 24.03 | 22.0 | Title Case (69.6%), Mixed (30.4%) |
| `train_source2` | 2 | 104 | 25.10 | 23.0 | Title (46.7%), Mixed (28.6%), Upper (18.9%), Lower (5.8%) |
| `train_source3` | 2 | 123 | 25.20 | 23.0 | Title (61.0%), Mixed (29.8%), Lower (6.2%), Upper (3.0%) |
| `test_source1` | 3 | 92 | 23.84 | 22.0 | Title Case (65.5%), Mixed (34.5%), Upper (0.0%) |
| `test_source2` | 2 | 102 | 25.70 | 24.0 | Title (44.4%), Mixed (33.1%), Upper (17.5%), Lower (5.0%) |
| `test_source3` | 2 | 103 | 25.66 | 24.0 | Title (58.5%), Mixed (32.8%), Lower (5.5%), Upper (3.2%) |

### Concrete Name Variations Discovered in Ground Truth Matches
Through direct pairing of Source 1 entities with their ground truth matches, several systematic name variations were uncovered:

1. **Typographical & OCR Mutations:**
   - S1: `Maure Williams Colombier Inc`
   - Matched S2: `Maure Wilblims Colombier Inc` *(insertion: "wilblims")*
   - S1: `Dahlia Power Reliable Scientific LLC`
   - Matched S3: `Dahlia Ponr Reliable Scientific LLC` *(character substitution: "Ponr" vs "Power")*
   - S1: `Payne Enterprises`
   - Matched S2/S3: `Payne Enterpires`, `PAYNE-ENRTPRMISES`, `Payne Etrepndiels`

2. **Multilingual Script Transliteration & Mixing:**
   - **Tamil:**
     - S1: `Raj Investments LLP`
     - Matched S2: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` *(Exact Tamil transliteration)*
     - Matched S3: `Raj Investments எல்எல்பி` *(Mixed English Latin + Tamil legal entity suffix)*
   - **Hindi / Devanagari:**
     - S1: `Ss Food Private Limited`
     - Matched S2: `एसएस फूड प्राइवेट लिमिटेड`
     - S1: `Hotel Enterprises Limited`
     - Matched S2: `होटल एंटरप्राइजेज लिमिटेड`
   - **Gujarati, Telugu, Malayalam, Punjabi:**
     - Found across India records in Source 2 and Source 3 (e.g. `కృష్ణా ఇంపেক্স లిమిటెడ్`, `ૐ Foundation પ્રાઇવેટ લિમિટેડ`, `ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ ਪ੍ਰਾ. ਲਿ.`, `സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ്`).

3. **Legal Entity Suffix Permutations & Prefixes:**
   - Suffix moves to prefix:
     - S1: `Crystal Staffing Solutions LLC` $\rightarrow$ Matched S2: `LLC Crystal Sttfrifng Solutions`
     - S1: `Dick Regional Armada Corp` $\rightarrow$ Matched S3: `[Corp] Dick Regional Armada`
   - Legal token bracket noise:
     - `Obsidian, [[LLC]]`, `(Ltd) Producer Aim Solutions`, `[INCORPORATED] PEAK TRADIN6 NETWORKS`

4. **Web Domains & DBAs as Business Names:**
   - S1: `Maure Williams Colombier Inc` $\rightarrow$ Matched S3: `maurewilliamscolombier.com`
   - S1: `Obsidian, LLC` $\rightarrow$ Matched S3: `Korbrixx D.B.A. Obsidian, LLC`
   - S1: `Chordia & Partners` $\rightarrow$ Matched S2: `Chordia + Pagnters - 7306204978` *(Phone number embedded)*

5. **Accented Character Injection:**
   - `Payne Enterprises` $\rightarrow$ `Payne Énterprises`
   - `Lumay Boral` $\rightarrow$ `Lumay Bóral`
   - `Moncada Learning Center` $\rightarrow$ `Moncada Léarning Center`

---

## 9. Address Analysis

### Length Statistics
| Dataset | Min Len | Max Len | Mean Len | Median Len | Has Digits (%) | Has Non-ASCII (%) |
|:---|---:|---:|---:|---:|---:|---:|
| `train_source1` | 11 | 256 | 52.07 | 45.0 | 96.51% | 0.03% |
| `train_source2` | 0 | 249 | 46.23 | 41.0 | 90.65% | 9.50% |
| `train_source3` | 0 | 240 | 46.71 | 42.0 | 90.80% | 9.02% |
| `test_source1` | 11 | 268 | 57.21 | 51.0 | 95.85% | 4.26% |
| `test_source2` | 0 | 269 | 50.41 | 45.0 | 92.67% | 14.75% |
| `test_source3` | 0 | 267 | 48.74 | 44.0 | 92.49% | 14.35% |

### Concrete Address Patterns Observed in Ground Truth Matches
1. **Component Inversion / Reordering:**
   - S1: `630 45th Terrace, Kansas City, MO`
   - Matched S2: `KANSAS CITY, MO, 630 45ND TERRACE, null` *(City/State placed before street)*
   - Matched S3: `Missouri, 630 45th Terrace, Kansas City` *(State placed first)*
2. **Abbreviation vs Full Word Standards:**
   - Standard US: `Street` $\leftrightarrow$ `ST` $\leftrightarrow$ `SAINT` (erroneously expanded!)
   - `Drive` $\leftrightarrow$ `DR`, `Avenue` $\leftrightarrow$ `AVE`, `Road` $\leftrightarrow$ `RD`
   - State abbreviations: `MO` $\leftrightarrow$ `Missouri`, `NC` $\leftrightarrow$ `North Carolina`, `IL` $\leftrightarrow$ `Illinois`, `WA` $\leftrightarrow$ `Washington`, `TN` $\leftrightarrow$ `Tamil Nadu`, `UP` $\leftrightarrow$ `Uttar Pradesh`, `KA` $\leftrightarrow$ `Karnataka`, `DL` $\leftrightarrow$ `Delhi`.
3. **Literal `<NULL>` / `null` Strings in Addresses:**
   - Discovered in Source 2: e.g. `33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA` and `KANSAS CITY, MO, 630 45ND TERRACE, null`.
4. **P.O. Box & Unit Additions:**
   - Source 2/3 frequently append or omit PO Box or unit numbers: `1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH` vs `1056 Belden Avenue, Akron, OH`.
5. **Completely Empty Addresses in Matches:**
   - In multiple valid ground truth matches, S2 or S3 has an empty address string (`""`). The matching decision in those cases relies entirely on business name similarity and country.

---

## 10. Data Quality Issues

| Issue Identified | Prevalence | Affected Files | Impact on Entity Resolution |
|:---|:---|:---|:---|
| **Missing Addresses** | 2.6% - 3.4% of rows | `train_source2`, `train_source3`, `test_source2`, `test_source3` | Pairwise address similarity cannot be computed; model must gracefully fall back to name+country signals. |
| **Transliterated Native Scripts** | 9% - 19% of Indian rows | `train_source2/3`, `test_source2/3` | Standard English string distance (Levenshtein, Jaro) yields 0 similarity between Latin and Indic scripts without transliteration/phonetic mapping. |
| **Noise Strings (`<NULL>`, `null`)** | Detected in address fields | Source 2 | Literal strings like `<NULL>` falsely match each other if not cleaned. |
| **Leading Punctuation / Formatting Artifacts** | 1.7% - 2.2% of names | Source 2 & Source 3 | Prefixes like `--`, `***`, `[[`, `@`, `#`, `(Ltd)` distort prefix-based blocking. |
| **Erroneous Expansion of Abbreviations** | Sporadic | Source 2 & Source 3 | `ST` in street addresses mistakenly expanded to `SAINT` instead of `STREET`. |
| **Phone Numbers / URLs in Names** | Sporadic | Source 2 & Source 3 | Substrings like `.com` or `- 7306204978` corrupt token overlap metrics unless sanitized. |
| **Unmatched Distractor Entities** | **25% - 27% of S2 and S3** | `train_source2`, `train_source3` (and test equivalents) | ~1.34 million entities in each source are distractors that do NOT match any Source 1 record. |

---

## 11. Important Observations for Entity Resolution

1. **Mapping Cardinality is Strictly 1-to-Many from Source 1 to {Source 2, Source 3}:**
   - No Source 2 or Source 3 entity is linked to more than one Source 1 entity (0 many-to-one collisions).
   - An individual Source 1 entity matches between **0 and 11 records** across Source 2 and Source 3 (average: ~3.5 matches for matching entities).
   - **89% of Source 1 entities have multiple matches**, meaning businesses appear in multiple variations/filings across the sources.

2. **Country as a Hard Blocking Criterion:**
   - Ground truth contains **zero cross-country matches**.
   - Because the country field is 100% populated in all files with 0 missing values, **country is a 100% safe, non-lossy blocking key**.
   - However, **country is open-set**: any country-based blocking must dynamically partition by `country` value rather than hardcoding specific country lists.

3. **High Proportion of Shared Names (38.3%):**
   - More than 845,000 Source 1 records share an exact name with another distinct business.
   - Therefore, candidate generation and scoring cannot rely on name alone; spatial/address alignment (city, state, postal code) is critical to prevent false clustering.

4. **Multi-Source Asymmetry:**
   - 80.5% of Source 1 entities match records in **both** Source 2 and Source 3.
   - 6.5% match S2 only, 7.5% match S3 only, and 5.6% match neither.

---

## 12. Recommendations for the Next Stage

Based strictly on empirical evidence from the 26.4M analyzed records:

1. **Text Normalization Engine (`src/normalization.py`):**
   - **Case normalization:** Lowercase all strings across all sources.
   - **Noise stripping:** Remove leading/trailing non-alphanumeric noise characters (`--`, `***`, `[[`, `]]`, `@`, `#`).
   - **Literal null handling:** Replace literal strings `<NULL>`, `null`, `None` with empty strings.
   - **Legal entity standardization:** Standardize and isolate company suffixes (`LLC`, `L.L.C.`, `Inc`, `Corp`, `Private Limited`, `Pvt Ltd`, `LLP`, `SARL`).
   - **Address token expansion:** Standardize bidirectional abbreviations (`St` $\leftrightarrow$ `Street`, `Ave` $\leftrightarrow$ `Avenue`, `Rd` $\leftrightarrow$ `Road`, `Dr` $\leftrightarrow$ `Drive`).
   - **Script transliteration:** Integrate Latin transliteration (e.g. Indic-to-Latin / Devanagari/Tamil transliteration or script-aware character n-grams) to bridge the 9-19% non-ASCII gap between S1 (pure Latin) and S2/S3 (native scripts).

2. **Blocking & Candidate Generation (`src/blocking.py`):**
   - **Tier 1 (Hard Block):** Partition by `country` (open-set categorical grouping). Never compare records from different countries.
   - **Tier 2 (Multi-pass Candidate Generation):**
     - Pass A: Standardized Name Prefix / Token Sorting Key + State/Region.
     - Pass B: Soundex / Metaphone / Character N-gram MinHash on normalized business names within the same country.
     - Pass C: Strict Address street/postal block (to capture cases where business names are drastically mutated e.g. DBA vs Legal name).

3. **Feature Engineering (`src/features.py`):**
   - Compute name similarity metrics: Levenshtein ratio, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, and longest common substring.
   - Compute address similarity metrics: address token intersection, postal code exact match, city/state match indicator.
   - Missing address indicator flag: explicitly model when S2/S3 address is empty so tree models adjust weight toward name similarity.

4. **Post-Processing & Constraints (`src/threshold.py`):**
   - Enforce the 1-to-many constraint: Since ground truth demonstrates that no S2 or S3 entity links to multiple S1 entities, any post-processing assignment algorithm (e.g. Hungarian algorithm, greedy matching, or bipartite graph matching) can resolve conflicting assignments.
