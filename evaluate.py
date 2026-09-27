"""Score SemIf predictions against PubMed RCT labels."""

import argparse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel, computed_field

from records import LABELS, DecisionRow, Label, Prediction, Record, read_jsonl


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


class LabelMetrics(BaseModel):
    label: Label
    true_positives: int
    predicted: int
    support: int

    @computed_field
    @property
    def precision(self) -> float:
        return _ratio(self.true_positives, self.predicted)

    @computed_field
    @property
    def recall(self) -> float:
        return _ratio(self.true_positives, self.support)

    @computed_field
    @property
    def f1(self) -> float:
        return _ratio(2 * self.precision * self.recall, self.precision + self.recall)


class ConfusionMatrix(Record):
    """Counts keyed by (gold, predicted) label pairs."""

    counts: dict[tuple[Label, Label], int]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[Label, Label]]) -> Self:
        return cls(counts=Counter(pairs))

    def count(self, gold: Label, predicted: Label) -> int:
        return self.counts.get((gold, predicted), 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accuracy(self) -> float:
        return _ratio(sum(self.count(label, label) for label in LABELS), self.total)

    def metrics(self, label: Label) -> LabelMetrics:
        return LabelMetrics(
            label=label,
            true_positives=self.count(label, label),
            predicted=sum(self.count(gold, label) for gold in LABELS),
            support=sum(self.count(label, predicted) for predicted in LABELS),
        )

    @property
    def macro_f1(self) -> float:
        return sum(self.metrics(label).f1 for label in LABELS) / len(LABELS)


def pair_labels(rows: list[DecisionRow], predictions: list[Prediction]) -> list[tuple[Label, Label]]:
    gold = {row.id: row.label for row in rows}
    return [(gold[prediction.id], prediction.choice) for prediction in predictions]


def format_report(matrix: ConfusionMatrix, expected_rows: int) -> str:
    lines = [
        f"Scored {matrix.total}/{expected_rows} rows   accuracy {matrix.accuracy:.3f}",
        "",
        f"{'label':<12}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}",
    ]
    for m in map(matrix.metrics, LABELS):
        lines.append(f"{m.label:<12}{m.precision:>7.3f}{m.recall:>8.3f}{m.f1:>7.3f}{m.support:>9}")
    lines.append(f"{'macro':<12}{'':>15}{matrix.macro_f1:>7.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Rows written by prepare.py")
    parser.add_argument("--predictions", type=Path, required=True, help="semif-score output")
    args = parser.parse_args()

    rows = read_jsonl(args.input, DecisionRow)
    matrix = ConfusionMatrix.from_pairs(pair_labels(rows, read_jsonl(args.predictions, Prediction)))
    print(format_report(matrix, len(rows)))


if __name__ == "__main__":
    main()
