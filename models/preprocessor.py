from typing import Sequence, Tuple

from models.article import Article


LABELS = {"propaganda": 1,
          "non-propaganda": 0}
INT_LABELS = {1: "propaganda",
              0: "non-propaganda"}


class DataPreprocessor:
    """ Pre-processes the data with modifications requires by the model.

    Example of modifications are to_lowercase, tokenization etc.
    """

    def __init__(self):
        pass

    # TODO - preprocessings
    #  - lowercase maybe?
    #  - punctuation;
    #  - common words (isn't -> is not)?
    def preprocess(self, articles: Sequence[Article]) -> Sequence[Article]:
        return articles

    def prepare(self, articles: Sequence[Article]) -> Sequence[Tuple[str, int]]:
        sentences = []
        labels = []
        for article in articles:
            [sentences.append(s) for s in article.article_sentences]
            [labels.append(l) for l in article.slc_labels]

        int_labels = map(self.label_to_int, labels)
        return [(sentence, label)
                for sentence, label in zip(sentences, int_labels)]

    @staticmethod
    def label_to_int(label: str):
        if label in LABELS:
            return LABELS.get(label)
        else:
            raise Exception("Label ({}) is not recognized!"
                            .format(label))

    @staticmethod
    def int_to_label(int_label: int):
        if int_label in INT_LABELS:
            return INT_LABELS.get(int_label)
        else:
            raise Exception("Int label ({}) is not recognized"
                            .format(int_label))