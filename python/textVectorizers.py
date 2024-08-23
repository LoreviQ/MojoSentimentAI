"""
Contains the text vectorizers used in the project
"""

import numpy as np


class CustomCountVectorizer:
    """
    Custom implementation of the CountVectorizer class.
    """

    def __init__(self, min_df=1, ngram_range=(1, 1)):
        self.vocabulary_ = {}
        self.min_df = min_df
        self.ngram_range = ngram_range

    def fit(self, documents):
        """
        Fit the vectorizer on the text data.
        """
        for document in documents:
            for word in document.split():
                if word in self.vocabulary_:
                    self.vocabulary_[word] = self.vocabulary_[word] + 1
                else:
                    self.vocabulary_[word] = 1
        for word, frequency in list(self.vocabulary_.items()):
            if frequency < self.min_df:
                del self.vocabulary_[word]
        self.vocabulary_ = {
            word: (i, frequency)
            for i, (word, frequency) in enumerate(self.vocabulary_.items())
        }

    def transform(self, documents):
        """
        Transform the text data into vectors.
        """
        vectors = []
        for document in documents:
            vector = np.zeros(len(self.vocabulary_), dtype=int)
            for word in document.split():
                if word in self.vocabulary_:
                    vector[self.vocabulary_[word][0]] += 1
            vectors.append(vector)
        return np.array(vectors)

    def fit_transform(self, documents):
        """
        Fit and transform the text data into vectors.
        """
        self.fit(documents)
        return self.transform(documents)
