# semif-bench

Classify sentences of medical abstracts with [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev):
a small open model (Qwen3.5-4B) reads a whole abstract and, for each sentence, picks one of the
five section labels. SemIf reads the model's probability for each option directly from a single
forward pass, so nothing is generated or parsed.

## About SemIf

SemIf (formerly OpenJev) is an independent open-source take on Jev, TypeSafe's closed service for
small runtime-defined decisions ("semantic ifs"). Instead of generating and parsing an answer, it
reads each option's score from one forward pass, which made it about 5× faster than generating a
JSON answer in SemIf's benchmark. Its probabilities aren't calibrated, so validate them on your task.

## Dataset

[PubMed RCT 20k](https://huggingface.co/datasets/armanc/pubmed-rct20k): about 20k abstracts of
randomized controlled trials, split into sentences. Each sentence is labeled with the heading it
appeared under: `background`, `objective`, `methods`, `results`, or `conclusions`.
Labels come from the authors' headings, so `background` and `objective` overlap heavily; most
abstracts use only one of the two.

## How it works

Each sentence becomes one SemIf decision row:

- **state**: the full abstract as numbered sentences
- **question**: which original heading sentence `[N]` appeared under
- **options**: the five labels, with descriptions of where their boundaries lie

All rows of an abstract share the same state, so SemIf's serial mode processes the abstract once
and reuses it for each of its sentences. The prompt wording lives in `prompt.py`.

| File | Purpose |
|---|---|
| `prepare.py` | Downloads a split and writes SemIf input rows |
| `prompt.py` | Question and option wording |
| `evaluate.py` | Accuracy, per-label precision/recall/F1, macro-F1 |
| `records.py` | Pydantic record types and JSONL I/O |
| `run.sh` | Prepare, score, and evaluate in one step |
| `example/` | Output of `./run.sh 5 0`: input rows and SemIf predictions (accuracy 0.848) |

## Setup

Requires an Apple Silicon Mac (SemIf's MLX backend) and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/westphal-jan/semif-bench.git
uv sync
```

The first run downloads the pinned model checkpoint (~9 GB) into the Hugging Face cache.

## Usage

```bash
./run.sh          # 5 random test abstracts, seed 0
./run.sh 20 1     # 20 abstracts, seed 1
```

Each run writes `input.jsonl`, `predictions.jsonl`, and `report.txt` to its own folder under
`results/`. The steps can also be run separately:

```bash
uv run python prepare.py --split dev --abstracts 40 --seed 0 --output input.jsonl
uv run semif-score --backend mlx --mode serial --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input input.jsonl --output predictions.jsonl
uv run python evaluate.py --input input.jsonl --predictions predictions.jsonl
```

Formatting and linting:

```bash
uv run ruff format . && uv run ruff check --fix .
```

## Results

Final prompt on 40 random test abstracts (438 sentences, `./run.sh 40 1`), Qwen3.5-4B in BF16 on an
Apple M3 Pro:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| background | 0.864 | 0.691 | 0.768 |
| objective | 0.694 | 0.833 | 0.758 |
| methods | 0.932 | 0.971 | 0.951 |
| results | 0.925 | 0.918 | 0.922 |
| conclusions | 0.924 | 0.924 | 0.924 |

**Accuracy 0.902, macro-F1 0.864**, at about 1.5 sentences/s. The largest group of errors, about a
third, is between `background` and `objective`, which depend on which heading the authors chose
rather than on the sentence itself.

## License

The code is MIT-licensed; see [LICENSE](LICENSE). The sentences in `example/` come from PubMed RCT,
which is built from PubMed abstracts and has no license of its own.
