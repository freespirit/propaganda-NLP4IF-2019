import logging
import os
import pandas as pd
import tqdm

from typing import Sequence

from models.article import Article
from models.articles_loader import ArticlesLoader
from models.cleaner import DataCleaner
from models.model import Model
from models.preprocessor import DataPreprocessor

DATASET_DIR = "datasets"
PROPAGANDA_MODEL_FILE = "propaganda.model"

TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "train-articles")
DEV_DATA_DIR = os.path.join(DATASET_DIR, "dev-articles")
TEST_DATA_DIR = os.path.join(DATASET_DIR, "test-articles")

TRAIN_LABELS_DIR_SLC = os.path.join(DATASET_DIR, "train-labels-SLC")
TRAIN_LABELS_DIR_FLC = os.path.join(DATASET_DIR, "train-labels-FLC")

ARTICLE_FILE_ID_PATTERN = "article(\\d*)\\.txt"
ARTICLE_LABEL_PATTERN_FLC = "article{:s}.task-FLC.labels"
ARTICLE_LABEL_PATTERN_SLC = "article{:s}.task-SLC.labels"

TEMPLATE_DEV_SLC = os.path.join(DATASET_DIR, "dev.template-output-SLC.out")
TEMPLATE_TEST_SLC = os.path.join(DATASET_DIR, "test.template-output-SLC.out")
OUTPUT_SLC_TXT = "outputs/dev.slc.txt"
OUTPUT_SLC_TXT_TEST = "outputs/test.slc.txt"

logging.basicConfig(level=logging.INFO)


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

    output.to_csv(sep="\t", header=False, index=False,
                  path_or_buf=OUTPUT_SLC_TXT)


if __name__ == "__main__":
    train_data_loader = ArticlesLoader(TRAIN_DATA_DIR,
                                       ARTICLE_FILE_ID_PATTERN,
                                       ARTICLE_LABEL_PATTERN_SLC,
                                       ARTICLE_LABEL_PATTERN_FLC,
                                       TRAIN_LABELS_DIR_SLC,
                                       TRAIN_LABELS_DIR_FLC)
    preprocessor = DataPreprocessor()
    articles = train_data_loader.load_data()

    articles = DataCleaner().clean(articles)
    articles = preprocessor.preprocess(articles)
    train_sentences = preprocessor.prepare(articles)

    model = Model()
    model.train_slc(train_sentences)

    test_data_loader = ArticlesLoader(TEST_DATA_DIR,
                                      ARTICLE_FILE_ID_PATTERN,
                                      ARTICLE_LABEL_PATTERN_SLC,
                                      ARTICLE_LABEL_PATTERN_FLC)
    test_articles = test_data_loader.load_data()

    for article in tqdm.tqdm(test_articles):
        sentences = article.article_sentences
        predictions = model.predict_slc(sentences)
        article.set_slc_labels(predictions)

    save_slc_predictions(test_articles)

    # predictions = eval_model.predict_flc(dev_articles)
    # save_flc_predictions(predictions)
