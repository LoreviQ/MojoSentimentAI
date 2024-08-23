"""
Contains the text vectorizers used in the project
"""

import numpy as np


class CustomCountVectorizer:
    """
    Custom implementation of the CountVectorizer class.
    """

    def __init__(self):
        self.vocabulary_ = {}
        self.vocab_index_ = 0

    def fit(self, texts):
        """
        Fit the vectorizer on the text data.
        """
        for text in texts:
            for word in text.split():
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = self.vocab_index_
                    self.vocab_index_ += 1

    def transform(self, texts):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for text in texts:
            vector = np.zeros(len(self.vocabulary_), dtype=int)
            for word in text.split():
                if word in self.vocabulary_:
                    vector[self.vocabulary_[word]] += 1
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, texts):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(texts)
        return self.transform(texts)
