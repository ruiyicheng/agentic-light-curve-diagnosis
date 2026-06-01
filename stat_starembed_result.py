import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


FINAL_DONE_MARKER = "Final summary diagnosis:"


def _log_path_for_csv(log_root: Path, csv_path: str) -> Path:
    csv_stem = Path(str(csv_path)).stem
    return log_root / f"{csv_stem}_diagnosis.txt"


def _extract_final_summary(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8")
    marker_index = text.rfind(FINAL_DONE_MARKER)
    if marker_index < 0:
        return None

    json_start = text.find("{", marker_index)
    if json_start < 0:
        return None

    decoder = json.JSONDecoder()
    try:
        final_summary, _ = decoder.raw_decode(text[json_start:])
    except json.JSONDecodeError:
        return None

    return final_summary if isinstance(final_summary, dict) else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _split_candidate_text(value: Any) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    for delimiter in ("|", ";", ","):
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]

    return [text]


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_label(value: Any, class_id_to_name: dict[str, str]) -> str:
    text = str(value).strip()
    return class_id_to_name.get(text, text)


def _extract_predictions(final_summary: dict[str, Any] | None) -> list[str]:
    if not final_summary:
        return []

    predictions: list[str] = []
    for raw_value in _as_list(final_summary.get("most_likely_variable_type")):
        predictions.extend(_split_candidate_text(raw_value))

    for raw_value in _as_list(final_summary.get("other_possible_variable_types")):
        predictions.extend(_split_candidate_text(raw_value))

    return _dedupe_preserving_order(predictions)


def _label_in_predictions(label: str, predictions: list[str]) -> bool:
    label_key = label.casefold()
    return any(prediction.casefold() == label_key for prediction in predictions)


def _build_stat_rows(
    overall_df: pd.DataFrame,
    log_root: Path,
    class_id_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, row in overall_df.iterrows():
        csv_path = str(row["light_curve_path"])
        csvfilename = Path(csv_path).name
        label = (
            _normalize_label(row["class_name"], class_id_to_name)
            if "class_name" in row and pd.notna(row["class_name"])
            else _normalize_label(row["class_id"], class_id_to_name)
        )

        final_summary = _extract_final_summary(_log_path_for_csv(log_root, csv_path))
        predictions = _extract_predictions(final_summary)
        including_true = _label_in_predictions(label, predictions)

        rows.append(
            {
                "csvfilename": csvfilename,
                "including_true": including_true,
                "exclusive_true": including_true and len(predictions) == 1,
                "in_candidite": including_true,
                "label": label,
                "prediction": "|".join(predictions),
            }
        )

    return rows


def _plot_confusion_matrix(
    stat_df: pd.DataFrame,
    class_id_to_name: dict[str, str],
    output_path: Path,
) -> None:
    label_order = [name for name in class_id_to_name.values() if name in set(stat_df["label"])]
    extra_labels = sorted(set(stat_df["label"]) - set(label_order))
    label_order.extend(extra_labels)

    prediction_values: set[str] = set()
    has_empty_prediction = False
    for prediction in stat_df["prediction"].fillna(""):
        candidates = [
            candidate.strip()
            for candidate in str(prediction).split("|")
            if candidate.strip()
        ]
        if candidates:
            prediction_values.update(candidates)
        else:
            has_empty_prediction = True

    prediction_order = [
        name for name in class_id_to_name.values() if name in prediction_values
    ]
    prediction_order.extend(sorted(prediction_values - set(prediction_order)))

    if not prediction_order:
        prediction_order = ["<no prediction>"]
    elif has_empty_prediction:
        prediction_order.append("<no prediction>")

    matrix = pd.DataFrame(0, index=label_order, columns=prediction_order, dtype=int)
    for _, row in stat_df.iterrows():
        label = row["label"]
        predictions = [
            candidate.strip()
            for candidate in str(row["prediction"]).split("|")
            if candidate.strip()
        ]
        if not predictions:
            matrix.loc[label, "<no prediction>"] += 1
            continue
        for prediction in predictions:
            matrix.loc[label, prediction] += 1

    width = max(8, 0.7 * len(matrix.columns) + 3)
    height = max(5, 0.55 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted candidate")
    ax.set_ylabel("Ground truth")
    ax.set_title("Starembed candidate confusion matrix")

    max_value = int(matrix.to_numpy().max()) if matrix.size else 0
    threshold = max_value / 2 if max_value else 0
    for row_index, label in enumerate(matrix.index):
        for column_index, prediction in enumerate(matrix.columns):
            value = int(matrix.loc[label, prediction])
            if value == 0:
                continue
            color = "white" if value > threshold else "black"
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    AI = "Gemini3.5flash"
    CLASS_ID_TO_NAME = {
        "1": "EW",
        "2": "EA",
        "3": "Beta_Lyrae",
        "4": "RRab",
        "5": "RRc",
        "6": "RRd",
        "7": "Blazhko",
        "8": "RS CVn",
        "9": "ACEP",
        "10": "Cep-II",
        "11": "HADS",
        "12": "LADS",
        "13": "LPV",
        "14": "ELL",
        "15": "Hump",
        "16": "PCEB",
        "17": "EA_UP",
    }

    # This script is used for stat the results of run_starembed_dev.py
    overall_csv_path = "/home/rui/code/project/pipeline_agentic_light_curve/input/starembed_sample/val/manifest.csv"
    overall_df = pd.read_csv(overall_csv_path)
    log_root = Path("/home/rui/code/project/pipeline_agentic_light_curve/output/starembed_dev")
    out_path = Path(f"starembed_stat_{AI}.csv")
    confusion_matrix_path = Path(f"starembed_confusion_matrix_{AI}.png")
    # Each file would become:
    # {"csvfilename":str,"including_true":True if final diag include true class, may be one of | ; "exclusive_true":True if final diag only include true class; "in_candidite":True if true class is in candidate list; "label":str label class; "prediction":[final candidate list separated by |]}
    # This would lead to a csv with columns: csvfilename, including_true, exclusive_true, in_candidite, label, prediction
    # Also plot the confusion matrix. We need to note that since the prediction is a list, each row may sum up to different number. The row is the ground truth, while the prediction would be candidate. Since the number of class in label and prediction may be different, the confusion matrix is not a square matrix.
    # Save the plot as starembed_confusion_matrix_{AI}.png

    stat_rows = _build_stat_rows(overall_df, log_root, CLASS_ID_TO_NAME)
    stat_df = pd.DataFrame(
        stat_rows,
        columns=[
            "csvfilename",
            "including_true",
            "exclusive_true",
            "in_candidite",
            "label",
            "prediction",
        ],
    )
    stat_df.to_csv(out_path, index=False)
    _plot_confusion_matrix(stat_df, CLASS_ID_TO_NAME, confusion_matrix_path)

    print(f"Saved stat CSV to {out_path}")
    print(f"Saved confusion matrix to {confusion_matrix_path}")
