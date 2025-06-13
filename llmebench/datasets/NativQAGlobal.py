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
                # for us: washington, florida, michigan, texas, north_carolina, maschusetts, california, pensylvania, illinois, ohio, hawaii, and georgia 
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
                "georgia": {
                    "dev": "georgia/nativqa_dev.tsv",
                    "test": "georgia/nativqa_test.tsv",
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

                # for georgia
                "georgia": {
                    "dev": "georgia/nativqa_dev.tsv",
                    "test": "georgia/nativqa_test.tsv",
                },

                # for iraq
                "iraq": {
                    "dev": "Iraq_Baghdad/nativqa_dev.tsv",
                    "test": "Iraq_Baghdad/nativqa_test.tsv",
                },

                # for mauritania
                "mauritania": {
                    "dev": "Mauritania_Nouakchott/nativqa_dev.tsv",
                    "test": "Mauritania_Nouakchott/nativqa_test.tsv",
                },

                # for kuwait
                "kuwait": {
                    "dev": "Kuwait_Kuwait_City/nativqa_dev.tsv",
                    "test": "Kuwait_Kuwait_City/nativqa_test.tsv",
                },

                # for lebanon
                "lebanon": {
                    "dev": "Lebanon_Beirut/nativqa_dev.tsv",
                    "test": "Lebanon_Beirut/nativqa_test.tsv",
                },

                # for libya
                "libya": {
                    "dev": "Libya_Tripoli/nativqa_dev.tsv",
                    "test": "Libya_Tripoli/nativqa_test.tsv",
                },
            
                # for morocco
                "morocco": {
                    "dev": "Morocco_Rabat/nativqa_dev.tsv",
                    "test": "Morocco_Rabat/nativqa_test.tsv",
                },

                # for oman
                "oman": {
                    "dev": "Oman_Muscat/nativqa_dev.tsv",
                    "test": "Oman_Muscat/nativqa_test.tsv",
                },

                # for saudia arabia riyadh
                "saudi_arabia": {
                    "dev": "Saudi_Arabia_Riyadh/nativqa_dev.tsv",
                    "test": "Saudi_Arabia_Riyadh/nativqa_test.tsv",
                },

                # for uae abu dhabi
                "uae": {
                    "dev": "UAE_Abu_Dhabi/nativqa_dev.tsv",
                    "test": "UAE_Abu_Dhabi/nativqa_test.tsv",
                },

                # for algeria
                "algeria": {
                    "dev": "Algeria_Algiers/nativqa_dev.tsv",
                    "test": "Algeria_Algiers/nativqa_test.tsv",
                },

                # for egypt
                "egypt": {
                    "dev": "Egypt_Cairo/nativqa_dev.tsv",
                    "test": "Egypt_Cairo/nativqa_test.tsv",
                },

                # for bahrain
                "bahrain": {
                    "dev": "Bahrain_Manama/nativqa_dev.tsv",
                    "test": "Bahrain_Manama/nativqa_test.tsv",
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
