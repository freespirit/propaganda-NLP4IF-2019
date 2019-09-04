import os
from collections import defaultdict
from typing import Sequence

import pandas as pd
import regex as re

from models.article import Article
from tools.src.annotations import Annotations
from tools.src.annotation import Annotation


class ArticlesLoader:
    """ A simple class that loads the article files in a provided directory as
    articles.

    The articles are provided by the workshop organizers in separate files in a
    directory. Each article consists of title and content sentences written
    separately on new lines (each). The name of the file contains the id of the
    article.

    """

    def __init__(self, data_dir,
                 article_file_id_pattern,
                 article_label_pattern_slc, article_label_pattern_flc,
                 labels_dir_slc=None, labels_dir_flc=None):

        self.data_dir = data_dir
        self.labels_dir_slc = labels_dir_slc
        self.labels_dir_flc = labels_dir_flc
        self.article_file_id_pattern = article_file_id_pattern
        self.article_label_pattern_slc = article_label_pattern_slc
        self.article_label_pattern_flc = article_label_pattern_flc

    def load_data(self) -> Sequence[Article]:
        """ Loads all the articles from the files in the provided directory.

        Returns a list of Article objects
        """
        article_files = os.listdir(self.data_dir)
        articles = [self.__map_to_article(os.path.join(self.data_dir, article))
                    for article in article_files]

        load_slc_labels: bool = self.labels_dir_slc is not None
        load_flc_labels: bool = self.labels_dir_flc is not None

        if load_slc_labels:
            for article in articles:
                self.__load_slc_labels(article)

        if load_flc_labels:
            for article in articles:
                self.__load_flc_labels(article)

        print("{} articles loaded".format(len(articles)))
        return articles

    def __map_to_article(self, file_path) -> Article:
        """Helper method that constructs an Article object from an article
        file"""
        with open(file_path) as file:
            article_id = re \
                .search(self.article_file_id_pattern, file.name, 0) \
                .group(1)
            content = file.readlines()
            return Article(article_id, content)

    def __load_slc_labels(self, article: Article):
        file_name = os.path.join(self.labels_dir_slc,
                                 self.article_label_pattern_slc
                                 .format(article.article_id))

        with open(file_name, mode="r") as file:
            slc_labels = pd.read_csv(file, sep="\t", names=["article_id",
                                                            "sentence_id",
                                                            "technique"])
            article.slc_labels = slc_labels.technique.values

    def __load_flc_labels(self, article: Article):
        article_id = article.article_id
        # print("Loading flc annotations for {}".format(article_id))

        file_name = os.path.join(self.labels_dir_flc,
                                 self.article_label_pattern_flc
                                 .format(article_id))

        article_annotations = Annotations()
        article_annotations.load_annotation_list_from_file(file_name)

        if article_annotations.has_article(article_id):
            annotations = article_annotations.get_article_annotations(article_id)
            spans = annotations.get_article_annotations()
        else:
            spans = []

        # convert the article annotations to sentence annotations
        sentence_annotations = self.__convert_annotations(article, spans)
        article.set_flc_annotations(sentence_annotations)

    @staticmethod
    def __convert_annotations(article, spans):
        """
        Converts an article-based annotation to an annotation inside a sentence
        :param article:
        :param spans: list of article-wide spans. E.g. each span start and end
        position is based on the article length, across sentences.
        :return: list of spans covering the sentences of the article. Each entry
        in the list is a bound inside a sentence.
        """
        article_text = "".join(article.article_sentences)
        article_annotations = []

        for i, sent in enumerate(article.article_sentences):
            sent_start = article_text.find(sent)
            assert sent_start != -1

            sentence_annotations = []
            sent_end = sent_start + len(sent)
            for span in spans:
                span_start = span.get_start_offset()
                span_end = span.get_end_offset()

                span_starts_in_sentence = sent_start <= span_start < sent_end
                span_ends_in_sentence = span_start < sent_start < span_end <= sent_end

                if span_starts_in_sentence:
                    sentence_annotation_start = span_start - sent_start
                    sentence_annotation_end = min(sent_end, span_end) - sent_start
                    sentence_annotation = Annotation(span.get_label(),
                                                     sentence_annotation_start,
                                                     sentence_annotation_end)
                    sentence_annotations.append(sentence_annotation)
                    assert sentence_annotation_start <= sentence_annotation_end
                elif span_ends_in_sentence:
                    sentence_annotation_start = 0
                    sentence_annotation_end = min(sent_end, span_end) - sent_start
                    sentence_annotation = Annotation(span.get_label(),
                                                     sentence_annotation_start,
                                                     sentence_annotation_end)
                    sentence_annotations.append(sentence_annotation)
                    assert sentence_annotation_start <= sentence_annotation_end

            article_annotations.append(sentence_annotations)

        return article_annotations
