from collections.abc import Callable, Mapping

from . import (
    alpaca,
    arc,
    cnn_dailymail,
    coqa,
    glue,
    hellaswag,
    mmlu,
    orca,
    race,
    squad,
    tiny
)

from ..task_base import BaseInstructDataset, BaseChoiceInstructDataset

DIRECTORY_CHOICE: dict[str, Mapping[str, Callable[[str], BaseChoiceInstructDataset]]] = {
    'glue': glue.DIRECTORY,
    'hellaswag': hellaswag.DIRECTORY,
    'mmlu': mmlu.DIRECTORY,
    'race': race.DIRECTORY,
    'arc': arc.DIRECTORY,
}

DIRECTORY_EXTRACT: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'squad': squad.DIRECTORY,
}

DIRECTORY_SUMMARY: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'cnn_dailymail': cnn_dailymail.DIRECTORY,
}

DIRECTORY_GENERATIVE: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'alpaca': alpaca.DIRECTORY,
    'orca': orca.DIRECTORY,
    'tiny': tiny.DIRECTORY,
}

DIRECTORY_CONVERSATIONAL: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'coqa': coqa.DIRECTORY,
}

DIRECTORY_ALL: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    **DIRECTORY_CHOICE,
    **DIRECTORY_EXTRACT,
    **DIRECTORY_SUMMARY,
    **DIRECTORY_CONVERSATIONAL,
    **DIRECTORY_GENERATIVE,
}
