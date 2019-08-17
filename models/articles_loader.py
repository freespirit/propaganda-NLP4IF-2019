import os
from typing import Sequence

import pandas as pd
import regex as re

from article import Article


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
        pass