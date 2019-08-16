import logging
import numpy as np
import os
import pandas as pd
import regex as re
import sklearn.metrics
import torch
import tqdm

from torch.utils.data import \
    TensorDataset, DataLoader, \
    RandomSampler, SequentialSampler
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


logging.basicConfig(level=logging.INFO)

COLUMN_TEXT = "text"
COLUMN_LABEL = "label"

EPOCHS = 1
BATCH_SIZE = 32


class Model:
    def __init__(self, model: torch.nn.Module = None):
        if model is not None:
            self.model = model
        else:
            self.model = torch.hub.load('huggingface/pytorch-pretrained-BERT',
                                        'bertForSequenceClassification',
                                        'bert-base-cased',
                                        num_labels=2)

        self.tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT',
                                        'bertTokenizer', 'bert-base-cased',
                                        do_basic_tokenize=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Device used: {}".format(self.device))

    @staticmethod
    def load(file_from) -> 'Model':
        # TODO actually load anything from the file, if recognized
        return Model()

    @staticmethod
    def save(model: 'Model', file_to):
        pass

    def train_slc(self, data: Sequence[Tuple[str, int]]):
        """ Trains a model to recognize propaganda sentences

        :type data: list of tuples - (sentence , propaganda_technique)
        """
        head = data[0]
        assert isinstance(head[0], str)
        assert isinstance(head[1], int)

        print("Training with {} samples".format(len(data)))
        df = pd.DataFrame(data=data, columns=[COLUMN_TEXT, COLUMN_LABEL])
        print(df.head())

        tokens_tensor = self.tokenize_texts(df[COLUMN_TEXT].values)
        labels_tensor = torch.tensor(df[COLUMN_LABEL].values)

        dataset = TensorDataset(tokens_tensor, labels_tensor)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        train_size = int(0.9 * len(train_dataset))
        validation_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, validation_size])

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=BATCH_SIZE)
        validation_dataloader = DataLoader(validation_dataset,
                                           sampler=SequentialSampler(validation_dataset),
                                           batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset,
                                     sampler=SequentialSampler(test_dataset),
                                     batch_size=BATCH_SIZE)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=2e-5)

        for epoch in tqdm.tqdm(range(EPOCHS)):
            self.model.train()
            for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
                optimizer.zero_grad()

                input_tensor, labels_tensor = tuple(t.to(self.device)
                                                    for t in batch)
                outputs = self.model(input_tensor, labels=labels_tensor)
                loss = outputs[0]

                if step % BATCH_SIZE == 100:
                    print(loss.item())

                loss.backward()
                optimizer.step()

            self.model.eval()
            for step, batch in enumerate(tqdm.tqdm(validation_dataloader)):
                inputs, labels = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    logits = outputs[0]

                    _, indices = torch.max(logits, dim=1)

                    validation_accuracy = sklearn.metrics.accuracy_score(
                        labels.detach().cpu().numpy(),
                        indices.numpy())
                    print("Validation accuracy: {}".format(validation_accuracy))

        test_sentences = [t[0] for t in test_dataloader][0].numpy()
        test_labels = [t[1] for t in test_dataloader][0].numpy()

        test_predictions = self.predict_slc(test_sentences)
        print("Test"
              "\n\taccuracy: {:.6f}"
              "\n\tprecision: {:.6f}"
              "\n\trecall: {:.6f}"
              "\n\tf1: {:.6f}".format(
                                sklearn.metrics.accuracy_score(test_labels, test_predictions),
                                sklearn.metrics.precision_score(test_labels, test_predictions),
                                sklearn.metrics.recall_score(test_labels, test_predictions),
                                sklearn.metrics.f1_score(test_labels, test_predictions)))

        baseline_predictions = np.ones_like(test_labels)
        print("Baseline"
              "\n\taccuracy: {:.6f}"
              "\n\tprecision: {:.6f}"
              "\n\trecall: {:.6f}"
              "\n\tf1: {:.6f}".format(
                            sklearn.metrics.accuracy_score(test_labels, baseline_predictions),
                            sklearn.metrics.precision_score(test_labels, baseline_predictions),
                            sklearn.metrics.recall_score(test_labels, baseline_predictions),
                            sklearn.metrics.f1_score(test_labels, baseline_predictions)))



    def tokenize_texts(self, texts: np.ndarray):
        tokenizer = self.tokenizer

        # wrap with the BERT [CLS] and [SEP] tokens
        def format_fn(sentence: Sequence[str]) -> Sequence[str]:
            return "{} {} {}".format(
                tokenizer.cls_token, sentence, tokenizer.sep_token)

        texts = [format_fn(sentence) for sentence in texts]
        tokenized_text = [tokenizer.tokenize(t) for t in texts]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        indexed_tokens = [torch.tensor(x) for x in indexed_tokens]

        tokens_tensor = torch.nn.utils.rnn.pad_sequence(
            indexed_tokens,
            batch_first=True,
            padding_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))

        return tokens_tensor

    def predict_slc(self, sentences: np.ndarray) -> Sequence[int]:
        result = list()

        tokens_tensor = self.tokenize_texts(sentences)
        predict_dataset = TensorDataset(tokens_tensor)
        predict_dataloader = DataLoader(predict_dataset,
                                        sampler=SequentialSampler(
                                            predict_dataset),
                                        batch_size=BATCH_SIZE)
        self.model.eval()
        for i, batch in enumerate(tqdm.tqdm(predict_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(batch[0])
                logits = outputs[0]

            _, indices = torch.max(logits, dim=1)
            indices = indices.detach().cpu().numpy()
            print(indices)
            result.extend(indices)

        assert len(result) == len(sentences)

        return result

    def predict_flc(self, articles):
        return [article for article in articles]


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
            raise Exception("Label ({}) is not recognized!"
                            .format(label))

    @staticmethod
    def int_to_label(int_label: int):
        if int_label in INT_LABELS:
            return INT_LABELS.get(int_label)
        else:
            raise Exception("Int label ({}) is not recognized"
                            .format(int_label))


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

    dev_data_loader = ArticlesLoader(DEV_DATA_DIR,
                                     ARTICLE_FILE_ID_PATTERN,
                                     ARTICLE_LABEL_PATTERN_SLC,
                                     ARTICLE_LABEL_PATTERN_FLC)
    dev_articles = dev_data_loader.load_data()

    for article in dev_articles:
        sentences = article.article_sentences
        predictions = model.predict_slc(sentences)
        article.set_slc_labels(predictions)

    save_slc_predictions(dev_articles)

    # predictions = eval_model.predict_flc(dev_articles)
    # save_flc_predictions(predictions)
