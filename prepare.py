"""Convert PubMed RCT 20k sentences into SemIf decision rows."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from prompt import OPTIONS, build_question, format_sentence
from records import AbstractState, DecisionRow, Sentence, read_jsonl, write_jsonl

DATASET = "armanc/pubmed-rct20k"
DATASET_REVISION = "091aec1e2384a20b2b36eb96177755ca13dd0b42"


def download_split(split: str) -> Path:
    return Path(hf_hub_download(DATASET, f"{split}.jsonl", repo_type="dataset", revision=DATASET_REVISION))


def group_abstracts(sentences: list[Sentence]) -> dict[str, list[Sentence]]:
    """Map abstract ID to its sentences in reading order, with abstracts in ID order."""
    abstracts: dict[str, list[Sentence]] = defaultdict(list)
    for sentence in sorted(sentences, key=lambda s: (s.abstract_id, s.sentence_id)):
        abstracts[sentence.abstract_id].append(sentence)
    return dict(abstracts)


def sample_ids(ids: list[str], count: int | None, seed: int) -> list[str]:
    if count is None:
        return ids
    return sorted(random.Random(seed).sample(ids, count))


def build_rows(sentences: list[Sentence]) -> list[DecisionRow]:
    # Identical state across an abstract's rows lets serial mode prefill the abstract once.
    state = AbstractState(abstract=[format_sentence(s) for s in sentences])
    return [
        DecisionRow(
            id=f"{s.abstract_id}-{s.sentence_id}",
            label=s.label,
            state=state,
            question=build_question(s),
            options=OPTIONS,
        )
        for s in sentences
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--abstracts", type=int, help="Random sample of N abstracts (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    abstracts = group_abstracts(read_jsonl(download_split(args.split), Sentence))
    ids = sample_ids(list(abstracts), args.abstracts, args.seed)
    count = write_jsonl(args.output, (row for i in ids for row in build_rows(abstracts[i])))
    print(f"Wrote {count} rows from {len(ids)} abstracts to {args.output}")


if __name__ == "__main__":
    main()
