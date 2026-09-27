"""
Candidate generation and blocking module for Amazon Business Entity Resolution Challenge 2026.

Reduces pairwise matching search space from O(N^2) to a manageable set of candidate pairs
using open-set dynamic country partitioning and multi-pass blocking over normalized
entity attributes.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import pandas as pd

from src.normalization import add_normalized_features


def _ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the DataFrame contains all required normalized features.
    If any required column is missing, normalization.py is called.
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
    """
    Removes common web-domain extensions from a token.

    Examples:
        poolegoevoss.com -> poolegoevoss
        company.in       -> company
    """

    if not token:
        return ""

    tok = str(token).lower()

    for ext in [
        ".com",
        ".in",
        ".org",
        ".net",
        ".co",
        ".io",
        ".biz",
        ".info",
    ]:
        if tok.endswith(ext) and len(tok) > len(ext):
            return tok[: -len(ext)]

    return tok


def generate_candidate_pairs_for_country(
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates candidate pairs for Source 1 and a target source
    within one country partition.

    Blocking passes:

    Pass 1:
        Significant business-name token matching.

    Pass 2:
        Concatenated business-name token matching.
        Handles cases such as:
            CLI Taxable LLC
            clitaxable.com

    Pass 3:
        Exact normalized core business-name matching.

    Pass 4:
        Postal code + house/unit number matching.

    Pass 5:
        Postal code + 2+ address-token overlap.

    Pass 6:
        Primary house/unit number + name/address evidence.

    Pass 7:
        Rare/informative address-token blocking.

        This pass is particularly important for multilingual
        business names where the names use different scripts.

        Example:
            English business name
                ↓
            Hindi/Telugu/Odia/Bengali business name

        but both records contain the same distinctive address.

    Returns:
        DataFrame containing:

            source1_entity_id
            candidate_entity_id
    """

    if df_s1.empty or df_target.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

    # ---------------------------------------------------------
    # Identify ID columns
    # ---------------------------------------------------------

    s1_id_col = (
        "entity_id"
        if "entity_id" in df_s1.columns
        else df_s1.columns[0]
    )

    target_id_col = (
        "entity_id"
        if "entity_id" in df_target.columns
        else df_target.columns[0]
    )

    # ---------------------------------------------------------
    # Convert target to records
    # ---------------------------------------------------------

    target_records = df_target.to_dict("records")

    # ---------------------------------------------------------
    # Generic business-name tokens
    # ---------------------------------------------------------

    GENERIC_NOISE_TOKENS = {
        "private",
        "limited",
        "pvt",
        "ltd",
        "inc",
        "llc",
        "corp",
        "co",
        "and",
        "company",
    }

    # ---------------------------------------------------------
    # Generic address tokens
    #
    # These should NOT be used alone for blocking because they
    # occur in huge numbers of records.
    # ---------------------------------------------------------

    GENERIC_ADDRESS_TOKENS = {
        "road",
        "rd",
        "street",
        "st",
        "avenue",
        "ave",
        "lane",
        "ln",
        "drive",
        "dr",
        "boulevard",
        "blvd",
        "highway",
        "hwy",
        "way",
        "place",
        "pl",
        "building",
        "bldg",
        "floor",
        "fl",
        "flat",
        "apartment",
        "apt",
        "unit",
        "block",
        "near",
        "opposite",
        "opp",
        "behind",
        "beside",
        "next",
        "house",
        "no",
        "city",
        "district",
        "state",
        "county",
        "india",
        "usa",
        "us",
        "private",
        "limited",
        "pvt",
        "ltd",
        "inc",
        "llc",
        "corp",
        "company",
        "and",
    }

    # ---------------------------------------------------------
    # Indexes
    # ---------------------------------------------------------

    token_index: Dict[str, List[str]] = {}

    core_exact_index: Dict[str, List[str]] = {}

    postal_index: Dict[str, List[Dict]] = {}

    number_index: Dict[str, List[Dict]] = {}

    # New Pass 7 indexes
    address_token_index: Dict[str, List[str]] = {}

    address_token_frequency: Counter = Counter()

    # Direct metadata lookup.
    # This avoids repeatedly searching through postal/number indexes.
    target_meta_index: Dict[str, Dict] = {}

    # ---------------------------------------------------------
    # STEP 1
    # Calculate frequency of informative address tokens
    # ---------------------------------------------------------

    for rec in target_records:

        addr_tokens = rec.get(
            "address_tokens",
            [],
        )

        if not isinstance(addr_tokens, (list, tuple, set)):
            continue

        for token in set(addr_tokens):

            token = str(token).strip().lower()

            if (
                len(token) >= 4
                and token not in GENERIC_ADDRESS_TOKENS
            ):
                address_token_frequency[token] += 1

    # ---------------------------------------------------------
    # STEP 2
    # Build target indexes
    # ---------------------------------------------------------

    for rec in target_records:

        tid = rec[target_id_col]

        c_core = rec.get(
            "business_name_core",
            "",
        )

        c_tokens = rec.get(
            "business_name_tokens",
            [],
        )

        c_postal = rec.get(
            "postal_code",
            "",
        )

        c_numbers = rec.get(
            "address_numbers",
            [],
        )

        c_addr_tokens = rec.get(
            "address_tokens",
            [],
        )

        # Safety for missing/null values
        if not isinstance(c_tokens, (list, tuple, set)):
            c_tokens = []

        if not isinstance(c_numbers, (list, tuple, set)):
            c_numbers = []

        if not isinstance(c_addr_tokens, (list, tuple, set)):
            c_addr_tokens = []

        # Convert to sets once
        c_tokens_set = set(c_tokens)

        c_addr_tokens_set = set(
            str(x).strip().lower()
            for x in c_addr_tokens
        )

        c_numbers_set = set(
            str(x).strip().lower()
            for x in c_numbers
        )

        # -----------------------------------------------------
        # Pass 1
        # Business-name token index
        # -----------------------------------------------------

        for tok in c_tokens:

            clean_tok = _strip_web_domain(tok)

            if (
                len(clean_tok) >= 3
                and clean_tok not in GENERIC_NOISE_TOKENS
            ):
                token_index.setdefault(
                    clean_tok,
                    [],
                ).append(tid)

        # -----------------------------------------------------
        # Pass 2
        # Concatenated business-name index
        # -----------------------------------------------------

        non_legal_tokens = [
            t
            for t in c_tokens
            if t not in GENERIC_NOISE_TOKENS
        ]

        if non_legal_tokens:

            concat_core = "".join(
                _strip_web_domain(t)
                for t in non_legal_tokens
            )

            if len(concat_core) >= 4:

                token_index.setdefault(
                    concat_core,
                    [],
                ).append(tid)

        # -----------------------------------------------------
        # Pass 3
        # Exact core-name index
        # -----------------------------------------------------

        if c_core and len(str(c_core)) >= 3:

            core_exact_index.setdefault(
                str(c_core),
                [],
            ).append(tid)

        # -----------------------------------------------------
        # Metadata used by address-based passes
        # -----------------------------------------------------

        target_meta = {
            "id": tid,
            "tokens_set": c_tokens_set,
            "addr_tokens_set": c_addr_tokens_set,
            "numbers_set": c_numbers_set,
        }

        target_meta_index[tid] = target_meta

        # -----------------------------------------------------
        # Pass 4 & 5
        # Postal-code index
        # -----------------------------------------------------

        if c_postal:

            c_postal = str(c_postal).strip().lower()

            postal_index.setdefault(
                c_postal,
                [],
            ).append(target_meta)

        # -----------------------------------------------------
        # Pass 6
        # Primary address-number index
        # -----------------------------------------------------

        if c_numbers:

            first_num = str(
                c_numbers[0]
            ).strip().lower()

            if first_num:

                number_index.setdefault(
                    first_num,
                    [],
                ).append(target_meta)

        # -----------------------------------------------------
        # Pass 7
        # Rare/informative address-token index
        #
        # Frequency <= 100 prevents very common address words
        # from producing huge candidate sets.
        # -----------------------------------------------------

        for addr_token in c_addr_tokens_set:

            addr_token = str(
                addr_token
            ).strip().lower()

            frequency = address_token_frequency.get(
                addr_token,
                0,
            )

            if (
                len(addr_token) >= 4
                and addr_token not in GENERIC_ADDRESS_TOKENS
                and frequency <= 100
            ):

                address_token_index.setdefault(
                    addr_token,
                    [],
                ).append(tid)

    # ---------------------------------------------------------
    # Collect candidate pairs
    # ---------------------------------------------------------

    candidate_pairs: List[
        Tuple[str, str]
    ] = []

    # ---------------------------------------------------------
    # Process Source 1 records
    # ---------------------------------------------------------

    for s1_rec in df_s1.to_dict("records"):

        s1_id = s1_rec[s1_id_col]

        s1_core = s1_rec.get(
            "business_name_core",
            "",
        )

        s1_tokens = s1_rec.get(
            "business_name_tokens",
            [],
        )

        s1_postal = s1_rec.get(
            "postal_code",
            "",
        )

        s1_numbers = s1_rec.get(
            "address_numbers",
            [],
        )

        s1_addr_tokens = s1_rec.get(
            "address_tokens",
            [],
        )

        # Safety for missing/null values
        if not isinstance(s1_tokens, (list, tuple, set)):
            s1_tokens = []

        if not isinstance(s1_numbers, (list, tuple, set)):
            s1_numbers = []

        if not isinstance(s1_addr_tokens, (list, tuple, set)):
            s1_addr_tokens = []

        # Sets for fast intersection
        s1_tokens_set = set(s1_tokens)

        s1_addr_tokens_set = set(
            str(x).strip().lower()
            for x in s1_addr_tokens
        )

        s1_numbers_set = set(
            str(x).strip().lower()
            for x in s1_numbers
        )

        matched_targets: Set[str] = set()

        # =====================================================
        # PASS 1
        # Name Token Blocking
        # =====================================================

        for tok in s1_tokens:

            clean_tok = _strip_web_domain(tok)

            if (
                len(clean_tok) >= 3
                and clean_tok not in GENERIC_NOISE_TOKENS
                and clean_tok in token_index
            ):

                matched_targets.update(
                    token_index[clean_tok]
                )

        # =====================================================
        # PASS 2
        # Concatenated Name Blocking
        # =====================================================

        non_legal_s1 = [
            t
            for t in s1_tokens
            if t not in GENERIC_NOISE_TOKENS
        ]

        if non_legal_s1:

            concat_s1 = "".join(
                _strip_web_domain(t)
                for t in non_legal_s1
            )

            if (
                len(concat_s1) >= 4
                and concat_s1 in token_index
            ):

                matched_targets.update(
                    token_index[concat_s1]
                )

        # =====================================================
        # PASS 3
        # Exact Core Name
        # =====================================================

        if (
            s1_core
            and len(str(s1_core)) >= 3
            and str(s1_core) in core_exact_index
        ):

            matched_targets.update(
                core_exact_index[str(s1_core)]
            )

        # =====================================================
        # PASS 4 & 5
        # Postal Code + Address Evidence
        # =====================================================

        if s1_postal:

            s1_postal = str(
                s1_postal
            ).strip().lower()

            if s1_postal in postal_index:

                for t_meta in postal_index[s1_postal]:

                    # Route A:
                    # Postal + matching house/unit number
                    if (
                        s1_numbers_set
                        & t_meta["numbers_set"]
                    ):

                        matched_targets.add(
                            t_meta["id"]
                        )

                    # Route B:
                    # Postal + 2+ address tokens
                    elif (
                        len(
                            s1_addr_tokens_set
                            & t_meta["addr_tokens_set"]
                        )
                        >= 2
                    ):

                        matched_targets.add(
                            t_meta["id"]
                        )

        # =====================================================
        # PASS 6
        # Address Number + Name/Address Evidence
        # =====================================================

        if s1_numbers:

            first_num = str(
                s1_numbers[0]
            ).strip().lower()

            if (
                first_num
                and first_num in number_index
            ):

                for t_meta in number_index[first_num]:

                    name_overlap = (
                        s1_tokens_set
                        & t_meta["tokens_set"]
                    )

                    address_overlap = (
                        s1_addr_tokens_set
                        & t_meta["addr_tokens_set"]
                    )

                    if (
                        name_overlap
                        or len(address_overlap) >= 2
                    ):

                        matched_targets.add(
                            t_meta["id"]
                        )

        # =====================================================
        # PASS 7
        # Rare / Informative Address Token Blocking
        #
        # This is the new pass.
        #
        # It helps recover:
        #
        # English:
        #     Future Healthcare Pvt Ltd
        #
        # Hindi:
        #     फ्यूचर हेल्थकेयर प्रा. लि.
        #
        # when the addresses are the same.
        # =====================================================

        informative_s1_address_tokens = []

        for addr_token in s1_addr_tokens_set:

            addr_token = str(
                addr_token
            ).strip().lower()

            frequency = address_token_frequency.get(
                addr_token,
                0,
            )

            if (
                len(addr_token) >= 4
                and addr_token not in GENERIC_ADDRESS_TOKENS
                and frequency <= 100
            ):

                informative_s1_address_tokens.append(
                    addr_token
                )

        # Count how many informative address tokens each
        # target shares with this Source 1 record.

        address_candidate_counts: Dict[
            str, int
        ] = {}

        for addr_token in informative_s1_address_tokens:

            target_ids = address_token_index.get(
                addr_token,
                [],
            )

            for tid in target_ids:

                address_candidate_counts[tid] = (
                    address_candidate_counts.get(
                        tid,
                        0,
                    )
                    + 1
                )

        # Apply conservative evidence rules.
        #
        # Rule A:
        #   Same address number + at least one
        #   informative address token.
        #
        # Rule B:
        #   At least two informative address tokens.

        for tid, overlap_count in address_candidate_counts.items():

            target_meta = target_meta_index.get(tid)

            if target_meta is None:
                continue

            shared_numbers = (
                s1_numbers_set
                & target_meta["numbers_set"]
            )

            # Strong evidence:
            # same house/unit number + distinctive
            # address token.
            if (
                shared_numbers
                and overlap_count >= 1
            ):

                matched_targets.add(tid)

            # Strong evidence:
            # two distinctive address tokens.
            elif overlap_count >= 2:

                matched_targets.add(tid)

        # =====================================================
        # Store candidate pairs
        # =====================================================

        for tid in matched_targets:

            candidate_pairs.append(
                (
                    s1_id,
                    tid,
                )
            )

    # ---------------------------------------------------------
    # No candidates
    # ---------------------------------------------------------

    if not candidate_pairs:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

    # ---------------------------------------------------------
    # Create result DataFrame
    # ---------------------------------------------------------

    result = pd.DataFrame(
        candidate_pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    # Remove duplicates
    result = (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return result


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame,
    target_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main candidate-generation function.

    Performs dynamic country-based partitioning.

    Only countries appearing in both Source 1 and the target
    source are compared.

    The country set is NOT hard-coded, allowing the system to
    support countries appearing in the test data, including
    France.

    Args:
        df_s1:
            Source 1 DataFrame.

        df_target:
            Source 2 or Source 3 DataFrame.

        target_label:
            Optional label such as "S2" or "S3".

    Returns:
        DataFrame containing:

            source1_entity_id
            candidate_entity_id

        and optionally:

            target_source
    """

    # ---------------------------------------------------------
    # Ensure normalized features exist
    # ---------------------------------------------------------

    s1_norm = _ensure_normalized(
        df_s1
    )

    target_norm = _ensure_normalized(
        df_target
    )

    # ---------------------------------------------------------
    # Dynamic country discovery
    # ---------------------------------------------------------

    s1_countries = set(
        s1_norm[
            "country_normalized"
        ].dropna().unique()
    )

    target_countries = set(
        target_norm[
            "country_normalized"
        ].dropna().unique()
    )

    common_countries = (
        s1_countries
        & target_countries
    )

    # ---------------------------------------------------------
    # Process each country independently
    # ---------------------------------------------------------

    all_country_candidates: List[
        pd.DataFrame
    ] = []

    for country in common_countries:

        s1_sub = s1_norm[
            s1_norm[
                "country_normalized"
            ]
            == country
        ]

        target_sub = target_norm[
            target_norm[
                "country_normalized"
            ]
            == country
        ]

        country_candidates = (
            generate_candidate_pairs_for_country(
                s1_sub,
                target_sub,
            )
        )

        if not country_candidates.empty:

            all_country_candidates.append(
                country_candidates
            )

    # ---------------------------------------------------------
    # No candidates
    # ---------------------------------------------------------

    if not all_country_candidates:

        columns = [
            "source1_entity_id",
            "candidate_entity_id",
        ]

        if target_label:
            columns.append(
                "target_source"
            )

        return pd.DataFrame(
            columns=columns
        )

    # ---------------------------------------------------------
    # Combine countries
    # ---------------------------------------------------------

    final_df = pd.concat(
        all_country_candidates,
        ignore_index=True,
    )

    # Remove duplicate candidate pairs
    final_df = (
        final_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Optional target-source label
    # ---------------------------------------------------------

    if target_label:

        final_df[
            "target_source"
        ] = target_label

    return final_df