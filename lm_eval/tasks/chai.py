"""
TODO
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
TODO
"""


class ChaiBase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None
    prompt = "prompt"
    chosen = "chosen"
    rejected = "rejected"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

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
            "query": doc[self.prompt],
            "choices": [doc[self.chosen], doc[self.rejected]],
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


class ChaiDavinci(ChaiBase):
    VERSION = 0
    DATASET_PATH = "AlekseyKorshuk/lmeh-chai-davinci"
    DATASET_NAME = None


class ChaiSynthetic(ChaiDavinci):
    VERSION = 0
    DATASET_PATH = "AlekseyKorshuk/lmeh-chai-synthetic"
    DATASET_NAME = None


class ChaiDavinciVsLit(ChaiDavinci):
    VERSION = 0
    DATASET_PATH = "AlekseyKorshuk/lmeh-chai-davinci-vs-lit"
    DATASET_NAME = None
    prompt = "prompt"
    chosen = "davinci"
    rejected = "lit"
