"""
Contains the classifiers used in the project
"""

from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MyRandomForestClassifier:
    """
    Custom implementation of the RandomForestClassifier class.
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.trees = []

    def _bootstrap_sample(self, x, y):
        """
        Create a bootstrap sample of the data.
        """
        y = y.reset_index(drop=True)

        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return x[indices], y[indices]

    def _random_features(self, x):
        """
        Select a random subset of features.
        """
        n_features = x.shape[1]
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        features = np.random.choice(n_features, max_features, replace=False)
        return features

    def fit(self, x, y):
        """
        Fit the RandomForest model on the data.
        """
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                x, y = self._bootstrap_sample(x, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            features = self._random_features(x)
            tree.fit(x[:, features], y)
            self.trees.append((tree, features))

    def predict(self, x):
        """
        Make predictions using the RandomForest
        """
        tree_preds = np.array(
            [tree.predict(x[:, features]) for tree, features in self.trees]
        )
        majority_votes = [
            Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(x.shape[0])
        ]
        return np.array(majority_votes)

    def score(self, x, y):
        """
        Compute the accuracy of the model.
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy
