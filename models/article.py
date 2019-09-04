from typing import Dict, Sequence

from tools.src.annotation import Annotation


class Article:
    def __init__(self, article_id: str, article_sentences: list):
        self.article_id = article_id
        self.article_sentences = article_sentences
        self.slc_labels = []
        self.flc_annotations = None

    def get_title(self):
        return self.article_sentences[0]

    def set_slc_labels(self, labels: Sequence[int]):
        """ Update the sentence labels with propaganda technique

        :param labels: a list of propaganda_techniques, each item matches a
        sentence in article_sentences
        """
        assert len(labels) == len(self.article_sentences)
        self.slc_labels = labels

    def set_flc_annotations(self, annotation_spans: Sequence[Sequence[Annotation]]):
        assert len(annotation_spans) == len(self.article_sentences)
        self.flc_annotations = annotation_spans
        # for sent, spans in annotation_spans.items():
        #     for span in spans:
        #         print(span)

