"""
Contains the text vectorizers used in the project
"""

import numpy as np
from gensim.models import Word2Vec


class CustomCountVectorizer:
    """
    Custom implementation of the CountVectorizer class.
    """

    def __init__(self, min_df=1, ngram_range=(1, 1)):
        self.vocabulary_ = {}
        self.min_df = min_df
        self.ngram_range = ngram_range

    def _generate_ngrams(self, document):
        """
        Generate ngrams from the text data.
        """
        words = document.split()
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i : i + n]))
        return ngrams

    def fit(self, documents):
        """
        Fit the vectorizer on the text data.
        """
        for document in documents:
            for ngram in self._generate_ngrams(document):
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

    def transform(self, documents):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for document in documents:
            vector = np.zeros(len(self.vocabulary_), dtype=int)
            for ngram in document.split():
                if ngram in self.vocabulary_:
                    vector[self.vocabulary_[ngram][0]] += 1
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, documents):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(documents)
        return self.transform(documents)


class Word2Vectorizer:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, documents):
        self.model = Word2Vec(
            sentences=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

    def transform(self, documents):
        vectors = []
        for document in documents:
            vector = np.zeros(self.vector_size)
            count = 0
            for word in document.split():
                if word in self.model.wv:
                    vector += self.model.wv[word]
                    count += 1
            if count > 0:
                vector /= count
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
