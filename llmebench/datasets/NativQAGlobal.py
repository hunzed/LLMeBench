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
                # for us: washington, florida, michigan, texas, north_carolina, maschusetts, california, pensylvania, illinois, ohio, and hawaii, 
                "washington": {
                    "dev": "washington/nativqa_dev.tsv",
                    "test": "washington/nativqa_test.tsv",
                },
                "florida": {
                    "dev": "florida/nativqa_dev.tsv",
                    "test": "florida/nativqa_test.tsv",
                },
                "michigan": {
                    "dev": "michigan/nativqa_dev.tsv",
                    "test": "michigan/nativqa_test.tsv",
                },
                "texas": {
                    "dev": "texas/nativqa_dev.tsv",
                    "test": "texas/nativqa_test.tsv",
                },
                "north_carolina": {
                    "dev": "north_carolina/nativqa_dev.tsv",
                    "test": "north_carolina/nativqa_test.tsv",
                },
                "massachusetts": {
                    "dev": "massachusetts/nativqa_dev.tsv",
                    "test": "massachusetts/nativqa_test.tsv",
                },
                "california": {
                    "dev": "california/nativqa_dev.tsv",
                    "test": "california/nativqa_test.tsv",
                },
                "pennsylvania": {
                    "dev": "pennsylvania/nativqa_dev.tsv",
                    "test": "pennsylvania/nativqa_test.tsv",
                },
                "illinois": {
                    "dev": "illinois/nativqa_dev.tsv",
                    "test": "illinois/nativqa_test.tsv",
                },
                "hawaii": {
                    "dev": "hawaii/nativqa_dev.tsv",
                    "test": "hawaii/nativqa_test.tsv",
                },
                "ohio": {
                    "dev": "ohio/nativqa_dev.tsv",
                    "test": "ohio/nativqa_test.tsv",
                },
                
                # for canada: ontario, and quebec
                "ontario": {
                    "dev": "ontario/nativqa_dev.tsv",
                    "test": "ontario/nativqa_test.tsv",
                },
                "quebec": {
                    "dev": "quebec/nativqa_dev.tsv",
                    "test": "quebec/nativqa_test.tsv",
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
