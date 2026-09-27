"""
Entity-level threshold optimization for Business Entity Resolution.

Optimizes the probability threshold using Macro F0.5 on the
validation Source1 entities only.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
)

VALIDATION_PREDICTIONS_FILE = (
    PROJECT_ROOT / "output" / "training" / "validation_predictions.tsv"
)

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01


def load_ground_truth():
    """Load ground truth into S1 -> set of matched S2/S3 IDs."""

    df = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()

        if matched == "":
            truth[s1_id] = set()
        else:
            truth[s1_id] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

    return truth


def calculate_entity_f05(actual, predicted):
    """Calculate F0.5 for one Source1 entity."""

    actual = set(actual)
    predicted = set(predicted)

    # Correct singleton / empty prediction
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0, 1.0, 1.0

    # Actual singleton but model predicted nothing
    if len(actual) > 0 and len(predicted) == 0:
        return 0.0, 0.0, 0.0

    # Actual singleton but model predicted wrong entity/entities
    if len(actual) == 0 and len(predicted) > 0:
        return 0.0, 0.0, 0.0

    true_positive = len(actual & predicted)

    precision = true_positive / len(predicted)
    recall = true_positive / len(actual)

    if precision == 0.0 and recall == 0.0:
        f05 = 0.0
    else:
        beta_squared = 0.25

        f05 = (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )

    return precision, recall, f05


def evaluate_threshold(predictions, truth, threshold):
    """Evaluate one probability threshold."""

    predictions_at_threshold = predictions[
        predictions["prediction_probability"] >= threshold
    ]

    predicted_by_s1 = {}

    for s1_id, group in predictions_at_threshold.groupby(
        "source1_entity_id"
    ):
        predicted_by_s1[s1_id] = set(group["target_entity_id"])

    precisions = []
    recalls = []
    f05_scores = []

    for s1_id, actual_matches in truth.items():

        predicted_matches = predicted_by_s1.get(s1_id, set())

        precision, recall, f05 = calculate_entity_f05(
            actual_matches,
            predicted_matches,
        )

        precisions.append(precision)
        recalls.append(recall)
        f05_scores.append(f05)

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f05_scores) / len(f05_scores),
    )


def main():

    print("=" * 70)
    print("ENTITY-LEVEL THRESHOLD OPTIMIZATION")
    print("=" * 70)

    print("\nLoading validation predictions...")

    predictions = pd.read_csv(
        VALIDATION_PREDICTIONS_FILE,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "target_entity_id": str,
            "label": int,
            "source": str,
            "prediction_probability": float,
        },
    )

    print(f"Validation prediction rows: {len(predictions):,}")

    # IMPORTANT:
    # Only evaluate Source1 entities that actually belong to
    # the validation set. Do NOT evaluate the entire training GT.
    validation_entities = set(
        predictions["source1_entity_id"].unique()
    )

    print(
        f"Validation Source1 entities: "
        f"{len(validation_entities):,}"
    )

    print("\nLoading ground truth...")

    full_truth = load_ground_truth()

    # Restrict ground truth to validation entities.
    truth = {
        s1_id: full_truth.get(s1_id, set())
        for s1_id in validation_entities
    }

    print(
        f"Ground-truth entities used for optimization: "
        f"{len(truth):,}"
    )

    missing_truth = [
        s1_id
        for s1_id in validation_entities
        if s1_id not in full_truth
    ]

    if missing_truth:
        print(
            f"WARNING: {len(missing_truth):,} validation entities "
            f"were not found in ground truth."
        )

    print("\nTesting thresholds...")

    thresholds = []

    current = THRESHOLD_MIN

    while current <= THRESHOLD_MAX + 1e-9:
        thresholds.append(round(current, 2))
        current += THRESHOLD_STEP

    results = []

    for threshold in thresholds:

        precision, recall, f05 = evaluate_threshold(
            predictions,
            truth,
            threshold,
        )

        results.append(
            {
                "threshold": threshold,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f05": f05,
            }
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"F0.5 {f05:.4f}"
        )

    results_df = pd.DataFrame(results)

    # Highest F0.5.
    # If there is an exact tie, choose the higher threshold
    # because the competition metric is precision-heavy.
    results_df = results_df.sort_values(
        ["macro_f05", "threshold"],
        ascending=[False, False],
    )

    best = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Best validation threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Validation Macro Precision : "
        f"{best['macro_precision']:.6f}"
    )

    print(
        f"Validation Macro Recall    : "
        f"{best['macro_recall']:.6f}"
    )

    print(
        f"Validation Macro F0.5      : "
        f"{best['macro_f05']:.6f}"
    )

    print("\nTop 10 thresholds:")

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    output_file = (
        PROJECT_ROOT
        / "output"
        / "training"
        / "threshold_results.tsv"
    )

    results_df.sort_values("threshold").to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved threshold results to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()