from collections.abc import Callable, Mapping

from . import (
    alpaca,
    arc,
    baai,
    cnn_dailymail,
    coqa,
    drop,
    glue,
    gsm8k,
    hellaswag,
    help_steer,
    hh,
    mmlu,
    obqa,
    openhermes,
    orca,
    piqa,
    race,
    squad,
    sciq,
    super_glue,
    tasksource,
    tiny,
    truthful_qa,
    tulu,
    ultrachat,
    ultrafeedback,
    winogrande
)

from ..task_base import BaseInstructDataset, BaseChoiceInstructDataset, BaseSteerInstructDataset

DIRECTORY_CHOICE: dict[str, Mapping[str, Callable[[str], BaseChoiceInstructDataset]]] = {
    'arc': arc.DIRECTORY,
    'glue': glue.DIRECTORY,
    'hellaswag': hellaswag.DIRECTORY,
    'mmlu': mmlu.DIRECTORY,
    'obqa': obqa.DIRECTORY,
    'piqa': piqa.DIRECTORY,
    'race': race.DIRECTORY,
    'sciq': sciq.DIRECTORY,
    'super_glue': super_glue.DIRECTORY,
    'truthful_qa': truthful_qa.DIRECTORY,
    'winogrande': winogrande.DIRECTORY,
}

DIRECTORY_EXTRACT: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'squad': squad.DIRECTORY,
    'drop': drop.DIRECTORY,
}

DIRECTORY_SUMMARY: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'cnn_dailymail': cnn_dailymail.DIRECTORY,
}

DIRECTORY_GENERATIVE: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'alpaca': alpaca.DIRECTORY,
    'baai': baai.DIRECTORY,
    'gsm8k': gsm8k.DIRECTORY,
    'OpenHermes': openhermes.DIRECTORY,
    'orca': orca.DIRECTORY,
    'tiny': tiny.DIRECTORY,
    'tulu': tulu.DIRECTORY,
    'ultrafeedback': ultrafeedback.DIRECTORY,
    'tasksource': tasksource.DIRECTORY,
}

DIRECTORY_CONVERSATIONAL: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    'coqa': coqa.DIRECTORY,
    'hh': hh.DIRECTORY,
    'ultrachat': ultrachat.DIRECTORY,
}

DIRECTORY_STEER: dict[str, Mapping[str, Callable[[str], BaseSteerInstructDataset]]] = {
    'HelpSteer': help_steer.DIRECTORY,
}

DIRECTORY_ALL: dict[str, Mapping[str, Callable[[str], BaseInstructDataset]]] = {
    **DIRECTORY_CHOICE,
    **DIRECTORY_EXTRACT,
    **DIRECTORY_SUMMARY,
    **DIRECTORY_CONVERSATIONAL,
    **DIRECTORY_GENERATIVE,
    **DIRECTORY_STEER,
}
