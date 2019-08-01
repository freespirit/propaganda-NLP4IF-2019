import os
import pandas as pd

from models.Article import Article
from typing import Sequence, Tuple


class Model:
    def __init__(self):
        pass

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
        df = pd.DataFrame(data=data)
        print(df.head())
        pass

    def predict_slc(self, sentences: Sequence[str]) -> Sequence[int]:
        return [1 for s in sentences]

    def predict_flc(self, articles: Sequence[Article]) -> Sequence[Article]:
        return [article for article in articles]
