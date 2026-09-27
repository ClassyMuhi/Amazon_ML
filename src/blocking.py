"""
Candidate generation and blocking module for Amazon Business Entity Resolution Challenge 2026.

Reduces pairwise matching search space from O(N^2) to a manageable set of candidate pairs
using open-set dynamic country partitioning and multi-pass blocking over normalized entity attributes.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from src.normalization import add_normalized_features


def _ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the DataFrame contains derived normalized features from normalization.py.
    If required derived columns are missing, calls add_normalized_features(df).
    """
    required_cols = [
        "business_name_normalized",
        "business_name_core",
        "business_name_tokens",
        "business_address_normalized",
        "address_tokens",
        "address_numbers",
        "postal_code",
        "country_normalized",
    ]
    if not all(col in df.columns for col in required_cols):
        return add_normalized_features(df)
    return df


def _strip_web_domain(token: str) -> str:
    """Strips web TLD extensions (.com, .in, .org, .net, .co) from tokens."""
    if not token:
        return ""
    tok = token.lower()
    for ext in [".com", ".in", ".org", ".net", ".co", ".io", ".biz", ".info"]:
        if tok.endswith(ext) and len(tok) > len(ext):
            return tok[: -len(ext)]
    return tok


def generate_candidate_pairs_for_country(
    df_s1: pd.DataFrame, df_target: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates unique candidate pairs (source1_entity_id, candidate_entity_id)
    for records within a single country partition using multi-pass blocking.

    Blocking Passes (Combined via UNION):
    - Pass 1: Name Token Index (all significant core/name tokens & domain-stripped tokens)
    - Pass 2: Concatenated Core Name Index (handles joined DBA/web-domain names e.g. clitaxable.com)
    - Pass 3: Exact Core Business Name Index
    - Pass 4: Address Route A - Exact Postal Code + House/Unit Number match
    - Pass 5: Address Route B - Exact Postal Code + 2+ Street Token overlap (recovers native script matches)
    - Pass 6: Address Route C - Primary House/Unit Number + 2+ Street Token overlap (when postal is missing)

    Args:
        df_s1: Source 1 DataFrame for a specific country (normalized).
        df_target: Target (Source 2 or 3) DataFrame for the same country (normalized).

    Returns:
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id'].
    """
    if df_s1.empty or df_target.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    s1_id_col = "entity_id" if "entity_id" in df_s1.columns else df_s1.columns[0]
    target_id_col = (
        "entity_id" if "entity_id" in df_target.columns else df_target.columns[0]
    )

    target_records = df_target.to_dict("records")

    # Inverted indexes for O(1) target lookup
    token_index: Dict[str, List[str]] = {}
    core_exact_index: Dict[str, List[str]] = {}
    postal_index: Dict[str, List[Dict]] = {}
    number_index: Dict[str, List[Dict]] = {}

    # Noise tokens to avoid over-blocking on generic single-letter/suffix tokens
    GENERIC_NOISE_TOKENS = {"private", "limited", "pvt", "ltd", "inc", "llc", "corp", "co", "and"}

    for rec in target_records:
        tid = rec[target_id_col]
        c_core = rec.get("business_name_core", "")
        c_tokens = rec.get("business_name_tokens", [])
        c_postal = rec.get("postal_code", "")
        c_numbers = rec.get("address_numbers", [])
        c_addr_tokens = rec.get("address_tokens", [])

        c_tokens_set = set(c_tokens)
        c_addr_tokens_set = set(c_addr_tokens)
        c_numbers_set = set(c_numbers)

        # Pass 1 & 2: Token Indexing & Concatenated Name Indexing
        for tok in c_tokens:
            clean_tok = _strip_web_domain(tok)
            if len(clean_tok) >= 3 and clean_tok not in GENERIC_NOISE_TOKENS:
                token_index.setdefault(clean_tok, []).append(tid)

        # Concatenated core tokens (e.g. 'clitaxable' for 'clitaxable.com' or 'cli taxable')
        non_legal_tokens = [t for t in c_tokens if t not in GENERIC_NOISE_TOKENS]
        if non_legal_tokens:
            concat_core = "".join(_strip_web_domain(t) for t in non_legal_tokens)
            if len(concat_core) >= 4:
                token_index.setdefault(concat_core, []).append(tid)

        # Pass 3: Exact Core Name Index
        if c_core and len(c_core) >= 3:
            core_exact_index.setdefault(c_core, []).append(tid)

        # Target metadata payload for address-based passes
        target_meta = {
            "id": tid,
            "tokens_set": c_tokens_set,
            "addr_tokens_set": c_addr_tokens_set,
            "numbers_set": c_numbers_set,
        }

        # Pass 4 & 5: Postal Index
        if c_postal:
            postal_index.setdefault(c_postal, []).append(target_meta)

        # Pass 6: Primary Address Number Index
        if c_numbers:
            first_num = c_numbers[0]
            if len(first_num) >= 1:
                number_index.setdefault(first_num, []).append(target_meta)

    # Collect candidate pairs
    candidate_pairs: List[Tuple[str, str]] = []

    for s1_rec in df_s1.to_dict("records"):
        s1_id = s1_rec[s1_id_col]
        s1_core = s1_rec.get("business_name_core", "")
        s1_tokens = s1_rec.get("business_name_tokens", [])
        s1_postal = s1_rec.get("postal_code", "")
        s1_numbers = s1_rec.get("address_numbers", [])
        s1_addr_tokens = s1_rec.get("address_tokens", [])

        s1_tokens_set = set(s1_tokens)
        s1_addr_tokens_set = set(s1_addr_tokens)
        s1_numbers_set = set(s1_numbers)

        matched_targets: Set[str] = set()

        # Pass 1: Name Token Overlap & Domain Matching
        for tok in s1_tokens:
            clean_tok = _strip_web_domain(tok)
            if len(clean_tok) >= 3 and clean_tok not in GENERIC_NOISE_TOKENS and clean_tok in token_index:
                matched_targets.update(token_index[clean_tok])

        # Concatenated Core Name Matching (handles clitaxable.com vs CLI Taxable LLC)
        non_legal_s1 = [t for t in s1_tokens if t not in GENERIC_NOISE_TOKENS]
        if non_legal_s1:
            concat_s1 = "".join(_strip_web_domain(t) for t in non_legal_s1)
            if len(concat_s1) >= 4 and concat_s1 in token_index:
                matched_targets.update(token_index[concat_s1])

        # Pass 2: Exact Core Name Match
        if s1_core and len(s1_core) >= 3 and s1_core in core_exact_index:
            matched_targets.update(core_exact_index[s1_core])

        # Pass 3 & 4: Postal Code Address Route (Recovers Native Script & DBA Matches)
        if s1_postal and s1_postal in postal_index:
            for t_meta in postal_index[s1_postal]:
                # Route A: Postal Code + Matching House/Unit Number
                if s1_numbers_set & t_meta["numbers_set"]:
                    matched_targets.add(t_meta["id"])
                # Route B: Postal Code + 2+ Street Token Overlap
                elif len(s1_addr_tokens_set & t_meta["addr_tokens_set"]) >= 2:
                    matched_targets.add(t_meta["id"])

        # Pass 5: Address Number Match (Primary House/Unit Number)
        if s1_numbers:
            first_num = s1_numbers[0]
            if len(first_num) >= 1 and first_num in number_index:
                for t_meta in number_index[first_num]:
                    # Require name token overlap OR 2+ street token overlap to control block size
                    if (s1_tokens_set & t_meta["tokens_set"]) or (len(s1_addr_tokens_set & t_meta["addr_tokens_set"]) >= 2):
                        matched_targets.add(t_meta["id"])

        for tid in matched_targets:
            candidate_pairs.append((s1_id, tid))

    if not candidate_pairs:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    res_df = pd.DataFrame(
        candidate_pairs, columns=["source1_entity_id", "candidate_entity_id"]
    )
    return res_df.drop_duplicates().reset_index(drop=True)


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame,
    target_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main public function for candidate generation across Source 1 and a Target dataset
    (Source 2 or Source 3).

    Performs hard blocking on country_normalized in a dynamic open-set manner,
    handling any set of countries (e.g. US, India, France, etc.) present in the data.

    Args:
        df_s1: Source 1 DataFrame.
        df_target: Target (Source 2 or Source 3) DataFrame.
        target_label: Optional label ('S2' or 'S3') to tag candidates.

    Returns:
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id']
        (plus 'target_source' if target_label is specified).
    """
    s1_norm = _ensure_normalized(df_s1)
    target_norm = _ensure_normalized(df_target)

    s1_countries = set(s1_norm["country_normalized"].unique())
    target_countries = set(target_norm["country_normalized"].unique())
    common_countries = s1_countries & target_countries

    all_country_candidates: List[pd.DataFrame] = []

    for country in common_countries:
        s1_sub = s1_norm[s1_norm["country_normalized"] == country]
        target_sub = target_norm[target_norm["country_normalized"] == country]

        c_pairs = generate_candidate_pairs_for_country(s1_sub, target_sub)
        if not c_pairs.empty:
            all_country_candidates.append(c_pairs)

    if not all_country_candidates:
        cols = ["source1_entity_id", "candidate_entity_id"]
        if target_label:
            cols.append("target_source")
        return pd.DataFrame(columns=cols)

    final_df = pd.concat(all_country_candidates, ignore_index=True)
    final_df = final_df.drop_duplicates().reset_index(drop=True)

    if target_label:
        final_df["target_source"] = target_label

    return final_df
