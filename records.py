"""Record types and JSONL I/O shared by prepare.py and evaluate.py."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

Label = Literal["background", "objective", "methods", "results", "conclusions"]
LABELS: tuple[Label, ...] = get_args(Label)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Sentence(Record):
    """One row of the PubMed RCT dataset."""

    abstract_id: str
    sentence_id: int
    label: Label
    text: str


class Option(Record):
    id: Label
    description: str


class AbstractState(Record):
    abstract: list[str]


class DecisionRow(Record):
    """A semif-score input row. SemIf ignores `label`; it is kept for evaluation."""

    id: str
    label: Label
    state: AbstractState
    question: str
    options: list[Option]


class Prediction(Record):
    """The fields of a semif-score result needed for evaluation."""

    id: str
    option_ids: list[Label]
    probabilities: list[float]

    @property
    def choice(self) -> Label:
        return self.option_ids[self.probabilities.index(max(self.probabilities))]


def read_jsonl[RecordT: BaseModel](path: Path, model: type[RecordT]) -> list[RecordT]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as destination:
        for record in records:
            destination.write(record.model_dump_json() + "\n")
            count += 1
    return count
