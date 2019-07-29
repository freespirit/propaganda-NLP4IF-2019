from typing import Sequence


class Article:
    def __init__(self, article_id: str, article_sentences: list):
        self.article_id = article_id
        self.article_sentences = article_sentences
        self.slc_labels = []
        self.flc_labels = []

    def get_title(self):
        return self.article_sentences[0]

    def set_slc_labels(self, labels: Sequence[int]):
        """ Update the sentence labels with propaganda technique

        :param labels: a list of propaganda_techniques, each item matches a
        sentence in article_sentences
        """
        assert len(labels) == len(self.article_sentences)
        self.slc_labels = labels

    def set_flc_labels(self, labels: Sequence[int]):
        pass