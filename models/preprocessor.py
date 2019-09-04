import numpy as np

from collections import defaultdict
from models.article import Article
from pytorch_transformers import BertTokenizer
from tools.src.annotation import Annotation
from typing import Sequence, Tuple


LABELS = {"propaganda": 1,
          "non-propaganda": 0}
INT_LABELS = {1: "propaganda",
              0: "non-propaganda"}


class DataPreprocessor:
    """ Pre-processes the data with modifications requires by the model.

    Example of modifications are to_lowercase, tokenization etc.
    """

    def __init__(self, tokenizer: BertTokenizer, techniques: Sequence[str]):
        """
        :param tokenizer: the same tokenizer used to train a model later.
        :param techniques: list of strings naming propaganda techniques.
        Make sure to pre-pend with neutral technique, e.g. non-propaganda.
        """
        self.tokenizer = tokenizer
        self.techniques = techniques

        # tokens to be used later to mark the beginning of a span
        self.additional_technique_tokens = ["[{}]".format(t) for t in self.techniques]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.additional_technique_tokens})

    # TODO - preprocessings
    #  - lowercase maybe?
    #  - punctuation;
    #  - common words (isn't -> is not)?
    def preprocess(self, articles: Sequence[Article]) -> Sequence[Article]:
        return articles

    def prepare(self, articles: Sequence[Article]) -> Sequence[Tuple[str, int, Sequence[int]]]:
        sentences = []
        labels = []
        token_labels = []

        for article in articles:
            article_sents = article.article_sentences
            article_techniques = self.__make_article_technique_labels(article)

            [sentences.append(s) for s in article_sents]
            [labels.append(l) for l in article.slc_labels]
            [token_labels.append(t) for t in article_techniques]

        sequence_labels = map(self.label_to_int, labels)

        # noinspection PyTypeChecker
        return [(sentence, label, techniques)
                for sentence, label, techniques
                in zip(sentences, sequence_labels, token_labels)]

    def __make_article_technique_labels(self, article):
        article_techniques = []
        article_sentences = article.article_sentences
        flc_spans = article.flc_annotations
        for i, sent in enumerate(article_sentences):
            original_sentence = sent
            sentence_annotations = flc_spans[i]
            annotations = sorted(sentence_annotations,
                                 key=lambda x: x.get_start_offset())
            sent_techniques = []

            # mark each char of the sentence with a label
            char_labels = np.zeros((len(sent)), dtype=int)
            for span in annotations:
                span_start = span.get_start_offset()
                span_end = span.get_end_offset()
                span_label = self.technique_to_int(span.get_label())
                for i in range(span_start, span_end):
                    char_labels[i] = span_label

            # print(char_labels)

            # find the starting positions of the spans
            last_seen_label = -1
            new_label_positions = {}
            for i, label in enumerate(char_labels):
                if last_seen_label != label:
                    last_seen_label = label
                    new_label_positions[i] = label

            # insert the technique token. Make sure the tag is the same format as in self.additional_tokens
            keys = sorted(new_label_positions.keys(), reverse=True)
            for k in keys:
                label = new_label_positions[k]
                sent = "{}{}{}".format(sent[0:k],
                                       self.__technique_label_to_token(label),
                                       sent[k:])

            # mark each token with the label of the last seen technique token
            tokens = self.tokenizer.tokenize(sent)
            last_seen_label = 0
            for i, token in enumerate(tokens):
                if token in self.additional_technique_tokens:
                    last_seen_label = self.additional_technique_tokens.index(token)
                else:
                    sent_techniques.append(last_seen_label)

            # assert len(sent_techniques) == len(self.tokenizer.tokenize(original_sentence))

            # print(sent)
            # print(tokens)
            # print(sent_techniques)

            article_techniques.append(sent_techniques)
        return article_techniques

    def make_flc_annotations_from_technique_labels(self, tokens, labels):
        # find the starting positions of the spans
        last_seen_label = -1
        new_label_positions = {}
        for i, label in enumerate(labels):
            if last_seen_label != label:
                last_seen_label = label
                new_label_positions[i] = label

        # insert technique tokens at span start positions
        tokens = tokens.squeeze()
        keys = sorted(new_label_positions.keys(), reverse=True)
        for k in keys:
            label = new_label_positions[k]
            technique_token = self.__technique_label_to_token(label)
            technique_token_id = self.tokenizer.convert_tokens_to_ids(technique_token)
            tokens = np.insert(tokens, k, technique_token_id)

        # print(tokens)
        sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # mark each char with a label based on the last seen technique token
        last_seen_label = -1
        label_positions = {}
        char_labels = []
        for i in range(len(sentence)):
            for technique_token in self.additional_technique_tokens:
                if sentence[i:].startswith(technique_token):
                    last_seen_label = self.additional_technique_tokens.index(technique_token)
                    label_positions[i] = last_seen_label
            char_labels.append(last_seen_label)

        # remove the technique tokens from the char labels
        keys = sorted(label_positions.keys(), reverse=True)
        for k in keys:
            label = label_positions[k]
            prev_span_end = k
            next_span_start = k + len(self.additional_technique_tokens[label])
            arr = char_labels[:prev_span_end]
            arr.extend(char_labels[next_span_start:])
            char_labels = arr

        # construct spans from the levels
        spans = []
        last_seen_label = 0
        last_span_start = 0
        for i, label in enumerate(char_labels):
            if last_seen_label != label or i == len(char_labels)-1:
                if last_seen_label != 0: # ignore the fake 'None' spans - they are normal speech
                    last_span_end = i
                    span = Annotation(self.int_to_technique(last_seen_label), last_span_start, last_span_end)
                    spans.append(span)
                last_span_start = i
                last_seen_label = label

        return spans

    def __technique_label_to_token(self, label):
        return "[{}]".format(self.int_to_technique(label))

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

    def technique_to_int(self, technique: str):
        if technique in self.techniques:
            return self.techniques.index(technique)
        else:
            return 0

    def int_to_technique(self, index: int):
        try:
            return self.techniques[index]
        except IndexError:
            print("No technique for index {}".format(index))
            exit(1)