"""
Contains the text vectorizers used in the project
"""

import numpy as np
from gensim.models import Word2Vec


class MyCountVectorizer:
    """
    Custom implementation of the CountVectorizer class.
    """

    def __init__(self, min_df=1, ngram_range=(1, 1)):
        self.vocabulary_ = {}
        self.min_df = min_df
        self.ngram_range = ngram_range

    def _generate_ngrams(self, tokens):
        """
        Generate ngrams from the text data.
        """
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def fit(self, data):
        """
        Fit the vectorizer on the text data.
        """
        for tokens in data:
            for ngram in self._generate_ngrams(tokens):
                if ngram in self.vocabulary_:
                    self.vocabulary_[ngram] = self.vocabulary_[ngram] + 1
                else:
                    self.vocabulary_[ngram] = 1
        for ngram, frequency in list(self.vocabulary_.items()):
            if frequency < self.min_df:
                del self.vocabulary_[ngram]
        self.vocabulary_ = {
            ngram: (i, frequency)
            for i, (ngram, frequency) in enumerate(self.vocabulary_.items())
        }

    def transform(self, data):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for tokens in data:
            vector = np.zeros(len(self.vocabulary_), dtype=int)
            for ngram in self._generate_ngrams(tokens):
                if ngram in self.vocabulary_:
                    vector[self.vocabulary_[ngram][0]] += 1
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, data):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(data)
        return self.transform(data)


class MyWord2Vectorizer:
    """
    Custom implementation of the Word2Vec class.
    """

    def __init__(self, vector_size=100, window=5, min_count=1, workers=-1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, data):
        """
        Fit the Word2Vec model on the text data.
        """
        self.model = Word2Vec(
            sentences=data,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

    def transform(self, data):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for tokens in data:
            vector = np.zeros(self.vector_size)
            count = 0
            for token in tokens:
                if token in self.model.wv:
                    vector += self.model.wv[token]
                    count += 1
            if count > 0:
                vector /= count
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, data):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(data)
        return self.transform(data)
