"""
Data normalization module for the Amazon Business Entity Resolution Challenge 2026.

This module provides modular, deterministic, and reproducible text cleaning,
normalization, tokenization, number extraction, and entity feature enrichment
for business names, addresses, and country attributes across all data sources.
"""

import re
import unicodedata
from typing import Any, Iterable, List, Optional
import pandas as pd


# Configurable default list of legal and corporate suffixes.
# Sorted by word length / complexity so multi-word variants match before single words.
DEFAULT_LEGAL_SUFFIXES: List[str] = [
    # Multi-word legal forms
    "private limited",
    "pvt limited",
    "pvt ltd",
    "private ltd",
    "public limited company",
    "limited liability partnership",
    "limited liability company",
    "professional corporation",
    "holding company",
    "holdings limited",
    "holdings inc",
    "ventures private limited",
    "ventures limited",
    "producer company",
    # Single-word legal forms (English / International)
    "limited",
    "ltd",
    "pvt",
    "private",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "plc",
    "co",
    "company",
    "llp",
    "pc",
    "pllc",
    "lp",
    "sarl",
    "sas",
    "sa",
    "sci",
    "gmbh",
    "ag",
    "bv",
    "nv",
    "dba",
    # Indic transliterated legal entity tokens (Tamil, Hindi, Gujarati, Telugu, Malayalam)
    "எல்எல்பி",
    "பிரைவேட் லிமிடெட்",
    "பிரைவேட்",
    "லிமிடெட்",
    "प्राइवेट लिमिटेड",
    "प्राइवेट",
    "लिमिटेड",
    "प्रा लि",
    "प्रा. लि.",
    "प्रा.लि.",
    "एलएलपी",
    "પ્રાઇવેટ લિમિટેડ",
    "પ્રાઇવેટ",
    "લિમિટેડ",
    "ప్రైవేట్ లిమిటెడ్",
    "లిమిటెడ్",
    "ప్రైవేట్",
    "പ്രൈവറ്റ് ലിമിറ്റഡ്",
    "ലിമിറ്റഡ്",
]

# Common legal prefixes that appear at the beginning of names (e.g. '[Corp] Name' or 'LLC Name')
DEFAULT_LEGAL_PREFIXES: set = {
    "llc",
    "inc",
    "corp",
    "corporation",
    "incorporated",
    "ltd",
    "limited",
    "pvt",
    "private",
    "sarl",
    "sas",
    "sa",
    "sci",
}


def _strip_latin_combining_diacritics(text: str) -> str:
    """
    Decomposes Unicode and removes Latin combining diacritical marks (e.g., é -> e, à -> a)
    without stripping combining characters or vowels in native Indic scripts (Devanagari, Tamil, etc.).
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    filtered = []
    for char in decomposed:
        # Unicode range U+0300 to U+036F represents standard Combining Diacritical Marks
        if unicodedata.category(char) == "Mn" and "\u0300" <= char <= "\u036f":
            continue
        filtered.append(char)
    return unicodedata.normalize("NFC", "".join(filtered))


def normalize_text(text: Any) -> str:
    """
    General robust text normalization.

    - Handles null/NaN/None safely -> returns empty string "".
    - Normalizes Unicode and removes Latin diacritical accents.
    - Converts text to lowercase.
    - Normalizes common punctuation variations (& -> and, + -> and).
    - Removes literal noise strings (<null>, <NULL>, null, none).
    - Normalizes multiple spaces, tabs, and newlines.
    - Strips leading and trailing non-alphanumeric noise.
    - Preserves meaningful alphanumeric information including all numbers and Indic scripts.

    Args:
        text: Input string or object.

    Returns:
        Cleaned, normalized lowercase string.
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization and Latin accent stripping
    cleaned = _strip_latin_combining_diacritics(text)
    cleaned = cleaned.lower()

    # Filter out literal null / none strings observed in data sources
    cleaned = re.sub(r"\b<(?:null|none)>\b|\b(?:null|none)\b", " ", cleaned)

    # Standardize conjunctions
    cleaned = re.sub(r"&", " and ", cleaned)
    cleaned = re.sub(r"(?<=\s)\+(?=\s)", " and ", cleaned)

    # Strip noisy brackets and wrapping punctuation
    cleaned = re.sub(r"^[^\w\s\u0900-\u0D7F]+|[^\w\s\u0900-\u0D7F]+$", " ", cleaned)

    # Consolidate whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_business_name(name: Any) -> str:
    """
    Normalizes a business name for comparison and candidate generation.

    - Applies base text normalization.
    - Removes punctuation dots inside acronyms (e.g. 'P.V.T.' -> 'pvt', 'L.L.C.' -> 'llc').
    - Cleans noise symbols, quotes, brackets, and prefixes.
    - Replaces internal hyphens, underscores, slashes with spaces to segment words cleanly.
    - Preserves digits, letters, and unicode scripts (including Indic matras/combining marks).

    Args:
        name: Raw business name string.

    Returns:
        Normalized business name string.
    """
    base = normalize_text(name)
    if not base:
        return ""

    # Remove periods in abbreviations (e.g., 'p.v.t.' -> 'pvt', 'u.s.' -> 'us', 'co.' -> 'co')
    cleaned = re.sub(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b)", "", base)

    # Replace hyphens, underscores, slashes between words with space
    cleaned = re.sub(r"[-_/\\|+]+", " ", cleaned)

    # Replace any punctuation and symbol characters with space,
    # strictly preserving Letters (L), Numbers (N), and Marks (M: vowel signs in Indic scripts)
    filtered = []
    for char in cleaned:
        cat = unicodedata.category(char)
        if cat.startswith(("P", "S")):
            filtered.append(" ")
        else:
            filtered.append(char)
    cleaned = "".join(filtered)

    # Final whitespace cleanup
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_name_tokens(normalized_name: str) -> List[str]:
    """
    Tokenizes a normalized business name into a list of word tokens.

    Args:
        normalized_name: Normalized business name string.

    Returns:
        List of non-empty token strings.
    """
    if not normalized_name:
        return []
    return [token for token in normalized_name.split() if token]


def get_core_business_name(
    name: str, legal_suffixes: Optional[Iterable[str]] = None
) -> str:
    """
    Extracts a conservative core-name representation of a business name by removing
    standard corporate/legal entity suffixes (and isolated legal prefixes).

    - Conservative: Only clearly recognized corporate entity forms are removed.
    - Does NOT drop words if doing so would leave an empty name.
    - Suffix list is fully configurable.

    Args:
        name: Raw or normalized business name.
        legal_suffixes: Optional custom iterable of suffix strings.

    Returns:
        Core business name string without legal suffixes.
    """
    norm = normalize_business_name(name)
    if not norm:
        return ""

    suffixes = (
        list(legal_suffixes)
        if legal_suffixes is not None
        else DEFAULT_LEGAL_SUFFIXES
    )
    # Sort suffixes by descending length to match longest multi-word suffix first
    sorted_suffixes = sorted(suffixes, key=lambda s: len(s), reverse=True)

    core = norm
    changed = True
    # Iteratively strip legal suffixes from the tail
    while changed:
        changed = False
        for s in sorted_suffixes:
            s_clean = s.lower().strip()
            # Match suffix preceded by space at end of string
            pattern = r"(?:\s+" + re.escape(s_clean) + r")+$"
            candidate = re.sub(pattern, "", core).strip()
            if candidate and candidate != core:
                core = candidate
                changed = True
                break

    # Also handle isolated legal prefixes (e.g. 'llc crystal staffing' -> 'crystal staffing')
    words = core.split()
    if len(words) > 1 and words[0] in DEFAULT_LEGAL_PREFIXES:
        prefix_stripped = " ".join(words[1:]).strip()
        if prefix_stripped:
            core = prefix_stripped

    # If stripping leaves an empty string, fallback to normalized name to avoid information loss
    return core if core else norm


def normalize_address(address: Any) -> str:
    """
    Normalizes a business address for comparison and blocking.

    - Handles null/NaN/empty addresses safely -> returns empty string "".
    - Converts text to lowercase and strips Latin diacritical accents.
    - Replaces literal null values ('<null>', 'null', 'none').
    - Standardizes comma, period, and whitespace separators.
    - Preserves building numbers, house numbers, floor/unit numbers, and postal codes.
    - Preserves internal slashes and hyphens in plot/unit numbers (e.g. '18/479e', 'af-684').

    Args:
        address: Raw address string.

    Returns:
        Normalized address string.
    """
    base = normalize_text(address)
    if not base:
        return ""

    # Remove quotes, brackets, and extraneous noise symbols
    cleaned = re.sub(r"['\"`’‘\[\]\(\)\{\}<>#@~*^$]", " ", base)

    # Standardize comma spacing: ensure single comma followed by space
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    # Normalize duplicate commas
    cleaned = re.sub(r"(?:,\s*)+,", ",", cleaned)

    # Standardize hyphens that are not between alphanumeric characters
    cleaned = re.sub(r"(?<![a-zA-Z0-9])-|-(?![a-zA-Z0-9])", " ", cleaned)

    # Remove remaining punctuation except commas, hyphens in numbers, and slashes in plot numbers
    cleaned = re.sub(r"[^\w\s,\/\-]", " ", cleaned)

    # Clean leading/trailing commas and whitespace
    cleaned = re.sub(r"^[\s,]+|[\s,]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_address_tokens(normalized_address: str) -> List[str]:
    """
    Tokenizes a normalized address into individual alphanumeric component tokens.

    Args:
        normalized_address: Normalized address string.

    Returns:
        List of non-empty token strings.
    """
    if not normalized_address:
        return []
    # Split on whitespace, commas, and semicolons
    raw_tokens = re.split(r"[\s,;]+", normalized_address)
    return [token.strip(".-/") for token in raw_tokens if token.strip(".-/")]


def extract_numbers(address: Any) -> List[str]:
    """
    Extracts all meaningful numeric and alphanumeric code tokens containing digits
    from an address (e.g. house numbers, unit numbers, plot IDs, postal codes).

    Examples:
        '12, M.G. Road, Chennai - 600001' -> ['12', '600001']
        '6(29), C.I.T. Colony, 2Nd Main Road' -> ['6', '29', '2nd']
        '18/479E/3 Peringattil Tower' -> ['18/479e/3']

    Args:
        address: Raw or normalized address string.

    Returns:
        List of extracted numeric/alphanumeric tokens in lowercase.
    """
    if address is None or pd.isna(address):
        return []
    if not isinstance(address, str):
        address = str(address)
    if not address.strip():
        return []

    # Match tokens with optional slash/hyphen structure that contain at least one digit
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[\/\-][a-zA-Z0-9]+)*", address)
    nums = [tok.lower() for tok in raw_tokens if any(c.isdigit() for c in tok)]
    return nums


def extract_postal_code(
    address: Any, country: Optional[str] = None
) -> str:
    """
    Identifies and extracts a postal code or ZIP code pattern from an address.
    Supports international formats without restricting or hard-coding country assumptions.

    Detects:
    - 5-digit US ZIP with optional +4 extension (e.g. '90210', '90210-1234')
    - 6-digit Indian PIN codes (e.g. '600001', '530003')
    - 5-digit French/European postal codes (e.g. '75008', '33000')
    - Alphanumeric postal codes (e.g. 'SW1A 1AA', 'H3Z 2Y7')

    Distinguishes postal codes from leading street numbers (e.g. '33466 Warwick Hills').

    Args:
        address: Raw or normalized address string.
        country: Optional country string (not required, open-set).

    Returns:
        Identified postal code string, or empty string '' if none detected.
    """
    if address is None or pd.isna(address):
        return ""
    if not isinstance(address, str):
        address = str(address)
    addr = address.strip()
    if not addr:
        return ""

    # 1. 5-digit US ZIP+4 pattern (e.g. '90210-1234')
    m = re.search(r"\b(\d{5}-\d{4})\b", addr)
    if m:
        return m.group(1)

    # 2. 6-digit PIN code (e.g. '600001', '530003')
    # Standard PIN codes begin with non-zero digit
    m = re.search(r"\b([1-9]\d{5})\b", addr)
    if m:
        return m.group(1)

    # 3. Alphanumeric postal code (e.g. Canadian / UK 'A1A 1A1' or 'SW1A 1AA')
    m = re.search(r"\b([A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d)\b", addr)
    if m:
        return m.group(1).replace(" ", "").upper()

    # 4. 5-digit postal code preceded by state abbreviation or comma (e.g. 'TX 75201', 'Paris, 75008')
    m = re.search(r"(?:[A-Za-z]{2}|,)\s*(\d{5})\b", addr)
    if m:
        return m.group(1)

    # 5. 5-digit code located at the very end of the address
    m = re.search(r"\b(\d{5})$", addr)
    if m:
        return m.group(1)

    # 6. Fallback: 5-digit number that is not at the very start of the address
    # (prevents house numbers like '33466 Warwick Hills Road' from being misidentified)
    tokens = addr.split()
    for tok in tokens[1:]:
        clean_tok = tok.strip(".,;:()[]-")
        if re.fullmatch(r"\d{5}", clean_tok):
            return clean_tok

    return ""


def normalize_country(country: Any) -> str:
    """
    Normalizes country values safely and deterministically without hardcoding
    an exclusive list of allowed countries (fully open-set).

    - Strips whitespace.
    - Two- or three-letter codes are uppercased (e.g. 'us' -> 'US', 'in' -> 'IN', 'fr' -> 'FR').
    - Longer country names are title-cased (e.g. 'india' -> 'India', 'france' -> 'France').
    - Null/empty values safely return empty string "".

    Args:
        country: Raw country string.

    Returns:
        Clean, standardized country string.
    """
    if country is None or pd.isna(country):
        return ""
    if not isinstance(country, str):
        country = str(country)
    c_clean = country.strip()
    if not c_clean:
        return ""

    if len(c_clean) <= 3:
        return c_clean.upper()
    return c_clean.title()


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a pandas DataFrame with normalized, core, tokenized, and extracted
    representations while strictly preserving all original columns and values.

    Added derived columns:
    - business_name_normalized: Cleaned, lowercase, standardized name.
    - business_name_core: Name with corporate/legal suffixes removed.
    - business_name_tokens: List of individual name tokens.
    - business_address_normalized: Cleaned, lowercase, standardized address.
    - address_tokens: List of address word and number tokens.
    - address_numbers: List of numeric/alphanumeric house/unit/plot codes.
    - postal_code: Extracted postal or ZIP code string (or empty string '').
    - country_normalized: Standardized country string.

    Args:
        df: Input DataFrame containing at least business_name, business_address, country.

    Returns:
        A copy of df augmented with the new derived columns.
    """
    # Create a shallow copy to prevent modifying caller's DataFrame in-place
    res = df.copy()

    # Fast list-comprehension based transformations (2-4x faster than series.apply on large scale)
    names = df["business_name"].fillna("").tolist() if "business_name" in df.columns else [""] * len(df)
    addrs = df["business_address"].fillna("").tolist() if "business_address" in df.columns else [""] * len(df)
    countries = df["country"].fillna("").tolist() if "country" in df.columns else [""] * len(df)

    # 1. Business Name Normalization & Features
    norm_names = [normalize_business_name(n) for n in names]
    core_names = [get_core_business_name(n) for n in names]
    name_tokens = [get_name_tokens(nn) for nn in norm_names]

    res["business_name_normalized"] = norm_names
    res["business_name_core"] = core_names
    res["business_name_tokens"] = name_tokens

    # 2. Business Address Normalization & Features
    norm_addrs = [normalize_address(a) for a in addrs]
    addr_tokens = [get_address_tokens(na) for na in norm_addrs]
    addr_numbers = [extract_numbers(a) for a in addrs]
    postal_codes = [extract_postal_code(a) for a in addrs]

    res["business_address_normalized"] = norm_addrs
    res["address_tokens"] = addr_tokens
    res["address_numbers"] = addr_numbers
    res["postal_code"] = postal_codes

    # 3. Country Normalization
    res["country_normalized"] = [normalize_country(c) for c in countries]

    return res
