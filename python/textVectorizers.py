"""
Contains the text vectorizers used in the project
"""

import numpy as np


class CustomCountVectorizer:
    """
    Custom implementation of the CountVectorizer class.
    """

    def __init__(self, min_df=1):
        self.vocabulary_ = {}
        self.vocab_index_ = 0
        self.min_df = min_df

    def fit(self, documents):
        """
        Fit the vectorizer on the text data.
        """
        for document in documents:
            for word in document.split():
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = self.vocab_index_
                    self.vocab_index_ += 1

    def transform(self, documents):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for text in documents:
            vector = np.zeros(len(self.vocabulary_), dtype=int)
            for word in text.split():
                if word in self.vocabulary_:
                    vector[self.vocabulary_[word]] += 1
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, documents):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(documents)
        vectors = self.transform(documents)
        frequency = np.sum(vectors, axis=0)
        mask = frequency >= self.min_df
        return vectors * mask
