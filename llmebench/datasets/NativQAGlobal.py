import csv

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class NativQAGlobalDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(NativQAGlobalDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "data_id": "a unique question id",
            "input": {
                "question": "question to be answered",
                "length": "number of words in answer",
            },
            "label": "A long answer",
        }

    @staticmethod
    def metadata():
        return {
            "language": "multilingual",
            "citation": """
            citation text goes here
            """,
            "link": "",
            "license": "",
            "splits": {
                "washington": {
                    "dev": "washington/nativqa_dev.tsv",
                    "test": "washington/nativqa_test.tsv",
                },
                "default": [
                ],
            },
            "task_type": TaskType.Other,
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                id = row[0]
                question = row[3]
                answer = row[4]
                length = len(answer.split())
                data.append(
                    {
                        "data_id": id,
                        "input": {"question": question, "length": length},
                        "label": answer,
                    }
                )
        return data
