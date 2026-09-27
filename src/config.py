"""
Configuration settings for the Amazon Business Entity Resolution Challenge 2026.

Defines global directory paths, training configuration, and reproducibility
settings used throughout the entity resolution pipeline.
"""

from pathlib import Path


# ============================================================
# Project root directory
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Core directory paths
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"

TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_DIR = PROJECT_ROOT / "models"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ============================================================
# Training configuration
# ============================================================

# Number of Source1 entities used for the initial training
# experiment.
#
# We start with a manageable sample because the complete
# dataset contains millions of records.
TRAIN_S1_SAMPLE_SIZE = 10000


# Number of additional target records sampled from each
# target source during the initial training experiment.
#
# Ground-truth target records are always included separately.
TRAIN_TARGET_SAMPLE_SIZE = 100000


# Number of rows read at a time from the large TSV files.
#
# This prevents the complete 5M+ row target files from being
# loaded into memory at once.
TARGET_CHUNK_SIZE = 100000


# Maximum number of negative candidate pairs retained for
# each Source1 entity.
#
# These negatives are generated through blocking, so they
# are harder and more useful for training than completely
# random negatives.
MAX_NEGATIVES_PER_S1 = 20


# Random seed used throughout experiments.
#
# Keeping this fixed makes experiments reproducible.
RANDOM_SEED = 42


# ============================================================
# Training output paths
# ============================================================

# Directory for intermediate training artifacts.
TRAINING_CANDIDATES_DIR = OUTPUT_DIR / "training"


# Labeled training candidate pairs.
#
# Format:
#   source1_entity_id
#   target_entity_id
#   label
#   source
TRAINING_PAIRS_FILE = (
    TRAINING_CANDIDATES_DIR / "training_pairs.tsv"
)


# Pairwise feature matrix used by LightGBM.
#
# TSV is used initially to avoid requiring an additional
# Parquet dependency.
TRAINING_FEATURES_FILE = (
    TRAINING_CANDIDATES_DIR / "training_features.tsv"
)


# ============================================================
# Model output paths
# ============================================================

LIGHTGBM_MODEL_FILE = (
    MODEL_DIR / "lightgbm_matcher.txt"
)

LIGHTGBM_METADATA_FILE = (
    MODEL_DIR / "lightgbm_metadata.json"
)


# ============================================================
# Validation / threshold configuration
# ============================================================

# Fraction of Source1 entities reserved for validation.
#
# IMPORTANT:
# The split will be performed by Source1 entity rather than
# by individual candidate pair to prevent information leakage.
VALIDATION_SIZE = 0.20


# Default probability threshold.
#
# This is only a starting value. The final threshold MUST be
# selected using entity-level macro F0.5 on validation data.
DEFAULT_MATCH_THRESHOLD = 0.50


# Threshold search range for the first experiment.
THRESHOLD_MIN = 0.10

THRESHOLD_MAX = 0.95

THRESHOLD_STEP = 0.01


# ============================================================
# LightGBM configuration
# ============================================================

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


# Number of rounds without validation improvement before
# LightGBM stops training.
LIGHTGBM_EARLY_STOPPING_ROUNDS = 100