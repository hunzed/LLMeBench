import json
import re

from llmebench.datasets import NativQAGlobalDataset
from llmebench.models import AzureModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "",
        "scores": {},
    }


def config():
    return {
        "dataset": NativQAGlobalDataset,
        "task": MultiNativQATask,
        "model": AzureModel,
        "general_args": {"test_split": "palestine"},
    }


def prompt(input_sample):
    base_prompt = f"Your task is to answer questions in Arabic based on a given context.\nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer\nContext:{input_sample['context']}\nQuestion:{input_sample['question']}\nAnswer:"
    
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]

def post_process(response):
    return response["choices"][0]["message"]["content"]