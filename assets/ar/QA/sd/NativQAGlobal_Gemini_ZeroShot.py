import json
import re

from llmebench.datasets import NativQAGlobalDataset
from llmebench.models import GeminiModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Gemini",
        "description": "",
        "scores": {},
    }


def config():
    return {
        "dataset": NativQAGlobalDataset,
        "task": MultiNativQATask,
        "model": GeminiModel,
        "general_args": {"test_split": "sudan"},
    }



def prompt(input_sample):
    # Define the question prompt
    question_prompt = f"""
    Please use your expertise to answer the following Arabic question. Answer in Arabic. Please provide Answer only. No additional text. Answer should be limited to less or equal to {input_sample['length']} words.

    Question: {input_sample['question']}
    
    """

    # Define the assistant prompt
    assistant_prompt = """
    You are an Arabic AI assistant specialized in providing detailed and accurate answers across various fields. Your task is to deliver clear, concise, and relevant information. 
    """

    return [
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
        {
            "role": "user",
            "content": question_prompt,
        },
    ]

def post_process(response):
    content = response[0]["content"]["parts"][0]["text"]
    content = content.replace("\n", "").strip()
    if "```json" in content:
        # content = content.replace("```json", "").replace('```', '').replace("\n}", "}")
        # content = content.replace("{\n", "{").replace("\",\n", "\",")

        content = re.search(r"```json(.*)```", content).group(1)
    return content
    # return json.loads(content)["answer"]
    # response = json.loads(data)
    # answer = response["answer"]
    return answer
