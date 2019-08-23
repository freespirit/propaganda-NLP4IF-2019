from typing import Sequence

from models.article import Article


class DataCleaner:
    def __init__(self):
        pass

    def clean(self, articles: Sequence[Article]) -> Sequence[Article]:
        return articles