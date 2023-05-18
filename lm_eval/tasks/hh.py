"""
TODO
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
TODO
"""


class HH(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "AlekseyKorshuk/lmeh-hh"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["prompt"],
            "choices": [doc["chosen"], doc["rejected"]],
            "gold": 0,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]