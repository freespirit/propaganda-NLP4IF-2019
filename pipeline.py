import logging
import os
import pandas as pd
import tqdm

from typing import Sequence

from models.article import Article
from models.articles_loader import ArticlesLoader
from models.cleaner import DataCleaner
from models.model import Model, COLUMN_TEXT, COLUMN_LABEL, COLUMN_TECHNIQUES
from models.preprocessor import DataPreprocessor

from tools.src.propaganda_techniques import Propaganda_Techniques

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
OUTPUT_SLC_TXT_DEV = "outputs/dev.slc.txt"
OUTPUT_SLC_TXT_TEST = "outputs/test.slc.txt"
OUTPUT_FLC_TXT_TEST = "outputs/test.flc.txt"

logging.basicConfig(level=logging.INFO)


# noinspection PyShadowingNames
def save_slc_predictions(articles: Sequence[Article],
                         template_slc_file, output_file):
    output = pd.read_csv(template_slc_file, sep="\t", names=["article_id",
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
                  path_or_buf=output_file)


# noinspection PyShadowingNames
def save_flc_predictions(articles: Sequence[Article], output_file):
    output = pd.DataFrame()

    for article in articles:
        article_id = article.article_id
        for sentence_annotations in article.flc_annotations:
            for annotation in sentence_annotations:
                row = pd.DataFrame({"id": article_id,
                                    "technique": annotation.get_label(),
                                    "begin_offset": annotation.get_start_offset(),
                                    "end_offset": annotation.get_end_offset()},
                                   index=[0])
                output = output.append(row, ignore_index=True)

    output.to_csv(sep="\t", header=False, index=False, path_or_buf=output_file)


# noinspection PyShadowingNames
def assert_backtranslating_labels_to_flc_works(articles, preprocessor):
    for i, article in tqdm.tqdm(enumerate(articles)):
        # print("Testing flc spans for article {}".format(article.article_id))
        data = preprocessor.prepare([article])
        df = pd.DataFrame(data=data,
                          columns=[COLUMN_TEXT, COLUMN_LABEL, COLUMN_TECHNIQUES])

        backtranslated_spans = []
        for i, (sentence, technique_labels) in enumerate(zip(df[COLUMN_TEXT].values, df[COLUMN_TECHNIQUES].values)):
            texts = [sentence]
            backtranslated_spans.append(
                preprocessor.make_flc_annotations_from_technique_labels(
                    model.tokenize_texts(texts).detach().cpu().numpy(),
                    technique_labels[128:]))

        article_sentences_spans = article.flc_annotations
        number_of_found_sentences_spans = len(backtranslated_spans)
        number_of_article_sentences_spans = len(article_sentences_spans)

        assert number_of_found_sentences_spans == \
            number_of_article_sentences_spans


if __name__ == "__main__":
    model = Model()

    train_data_loader = ArticlesLoader(TRAIN_DATA_DIR,
                                       ARTICLE_FILE_ID_PATTERN,
                                       ARTICLE_LABEL_PATTERN_SLC,
                                       ARTICLE_LABEL_PATTERN_FLC,
                                       TRAIN_LABELS_DIR_SLC,
                                       TRAIN_LABELS_DIR_FLC)
    propaganda_techniques = Propaganda_Techniques().techniques
    propaganda_techniques.insert(0, None)

    preprocessor = DataPreprocessor(model.tokenizer, propaganda_techniques)
    articles = train_data_loader.load_data()

    # assert_backtranslating_labels_to_flc_works(articles, preprocessor)

    articles = DataCleaner().clean(articles)
    articles = preprocessor.preprocess(articles)
    train_data = preprocessor.prepare(articles)

    # model.train(train_data)

    test_data_loader = ArticlesLoader(TEST_DATA_DIR,
                                      ARTICLE_FILE_ID_PATTERN,
                                      ARTICLE_LABEL_PATTERN_SLC,
                                      ARTICLE_LABEL_PATTERN_FLC)
    test_articles = test_data_loader.load_data()

    for article in tqdm.tqdm(test_articles):
        sentences = article.article_sentences

        # predictions = model.predict_slc(sentences)
        # article.set_slc_labels(predictions)

        (tokens, predictions) = model.predict_flc(sentences)
        spans = []
        for t, p in list(zip(tokens, predictions)):
            new_spans = preprocessor\
                .make_flc_annotations_from_technique_labels(t, p)
            spans.append(new_spans)
        article.set_flc_annotations(spans)

    save_slc_predictions(test_articles, TEMPLATE_TEST_SLC, OUTPUT_SLC_TXT_TEST)
    save_flc_predictions(test_articles, OUTPUT_FLC_TXT_TEST)
