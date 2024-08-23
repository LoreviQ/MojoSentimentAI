"""
This module contains functions to train a random forest classifier model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


def train_test_split(df, ratio=0.2):
    """
    Split the dataset into training and testing sets.
    Converts the text data into a vector.

    Args:
        df (pd.DataFrame): The dataset to split.
        ratio (float): The ratio of the testing set.

    Returns:
        tuple: The training and testing sets.
    """
    test = df.sample(frac=ratio)
    train = df.drop(test.index)
    cv = CountVectorizer(ngram_range=(1, 2))
    x_test = cv.fit_transform(test["text"])
    y_test = test["label"]
    x_train = cv.transform(train["text"])
    y_train = train["label"]
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """
    Train a random forest classifier model using grid search.

    Args:
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    parameters = {
        "max_features": ["sqrt"],
        "n_estimators": [500, 1000, 1500],
        "max_depth": [5, 10, None],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2, 5, 10],
        "bootstrap": [True, False],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        parameters,
        cv=5,
        return_train_score=True,
        n_jobs=-1,
        verbose=5,
    )
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_
