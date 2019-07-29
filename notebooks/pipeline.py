import os
import pandas as pd
from typing import Sequence, Tuple

from models.Article import Article
from models.DataLoader import DataLoader
from models.Model import Model
from typing import Sequence, Tuple

DATASET_DIR = "../datasets"
PROPAGANDA_MODEL_FILE = "propaganda.model"

TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "train-articles")
DEV_DATA_DIR = os.path.join(DATASET_DIR, "dev-articles")

TRAIN_LABELS_DIR_SLC = os.path.join(DATASET_DIR, "train-labels-SLC")
TRAIN_LABELS_DIR_FLC = os.path.join(DATASET_DIR, "train-labels-FLC")

ARTICLE_FILE_ID_PATTERN = "article(\\d*)\\.txt"
ARTICLE_LABEL_PATTERN_FLC = "article{:s}.task-FLC.labels"
ARTICLE_LABEL_PATTERN_SLC = "article{:s}.task-SLC.labels"

TEMPLATE_DEV_SLC = os.path.join(DATASET_DIR, "dev.template-output-SLC.out")
OUTPUT_SLC_TXT = "dev.slc.txt"

LABELS = {"propaganda": 1,
          "non-propaganda": 0}
INT_LABELS = {1: "propaganda",
              0: "non-propaganda"}


class DataCleaner:
    def __init__(self):
        pass

    def clean(self, articles: Sequence[Article]) -> Sequence[Article]:
        return articles


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
            raise Exception("Label ({}) is not recognized!".format(label))

    @staticmethod
    def int_to_label(int_label: int):
        if int_label in INT_LABELS:
            return INT_LABELS.get(int_label)
        else:
            raise Exception("Int label ({}) is not recognized".format(int_label))



def save_slc_predictions(articles: Sequence[Article]):
    output = pd.read_csv(TEMPLATE_DEV_SLC, sep="\t", names=["article_id",
                                                            "sentence_id",
                                                            "label"])
    for article in articles:
        article_id = int(article.article_id)
        labels = map(lambda p: DataPreprocessor.int_to_label(p),
                     article.slc_labels)
        for (sentence_id, prediction) in enumerate(list(labels)):
            mask = (output["article_id"] == article_id) &\
                   (output["sentence_id"] == (sentence_id + 1))
            output.loc[mask, "label"] = prediction

    output.to_csv(sep="\t", header=False, index=False, path_or_buf=OUTPUT_SLC_TXT)


if __name__ == "__main__":
    train_data_loader = DataLoader(TRAIN_DATA_DIR,
                                   ARTICLE_FILE_ID_PATTERN,
                                   ARTICLE_LABEL_PATTERN_SLC,
                                   ARTICLE_LABEL_PATTERN_FLC,
                                   TRAIN_LABELS_DIR_SLC, TRAIN_LABELS_DIR_FLC)
    preprocessor = DataPreprocessor()
    articles = train_data_loader.load_data()

    articles = DataCleaner().clean(articles)
    articles = preprocessor.preprocess(articles)
    train_sentences = preprocessor.prepare(articles)

    model = Model()
    model.train_slc(train_sentences)
    model.save(PROPAGANDA_MODEL_FILE)

    dev_data_loader = DataLoader(DEV_DATA_DIR,
                                 ARTICLE_FILE_ID_PATTERN,
                                 ARTICLE_LABEL_PATTERN_SLC,
                                 ARTICLE_LABEL_PATTERN_FLC)
    dev_articles = dev_data_loader.load_data()

    eval_model = Model()
    eval_model.load(PROPAGANDA_MODEL_FILE)

    for article in dev_articles:
        sentences = article.article_sentences
        predictions = eval_model.predict_slc(sentences)
        article.set_slc_labels(predictions)

    save_slc_predictions(dev_articles)

    # predictions = eval_model.predict_flc(dev_articles)
    # save_flc_predictions(predictions)
