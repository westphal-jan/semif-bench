"""Prompt wording: how sentences, questions, and label options are presented to the model."""

from records import Option, Sentence

OPTIONS = [
    Option(
        id="background",
        description=(
            "BACKGROUND: context before the methods, such as disease burden, prior evidence, knowledge gaps, or "
            "rationale. When the opening spans several sentences, authors usually file all of them here, "
            "including a closing aim statement."
        ),
    ),
    Option(
        id="objective",
        description=(
            "OBJECTIVE: the aim of the study, e.g. 'To evaluate ...' or 'The aim of this study was ...'. "
            "A single opening sentence is usually filed here; in a longer opening, only a final aim statement is."
        ),
    ),
    Option(
        id="methods",
        description=(
            "METHODS: what was done or planned: design, setting, eligibility, interventions, outcome measures, "
            "and analyses."
        ),
    ),
    Option(
        id="results",
        description=(
            "RESULTS: what happened: how many participants were enrolled, randomized, or completed the study, "
            "their characteristics, and outcome findings with or without statistics."
        ),
    ),
    Option(
        id="conclusions",
        description=(
            "CONCLUSIONS: the closing sentences that interpret or summarize the main findings, state implications, "
            "or recommend further research, usually without detailed statistics."
        ),
    ),
]


def _number(sentence: Sentence) -> int:
    return sentence.sentence_id + 1


def format_sentence(sentence: Sentence) -> str:
    """One line of the abstract state; its [N] marker is what `build_question` refers to."""
    return f"[{_number(sentence)}] {sentence.text}"


def build_question(sentence: Sentence) -> str:
    number = _number(sentence)
    return (
        "The evidence is a randomized controlled trial abstract with its section headings removed. "
        "Sections are contiguous and appear in the order background, objective, methods, results, conclusions; "
        "many abstracts use only one of background or objective. "
        f"Under which original heading did sentence [{number}] appear? "
        f'Sentence [{number}]: "{sentence.text}"'
    )
