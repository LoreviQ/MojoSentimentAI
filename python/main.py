"""
Main module to load the dataset and train the model.
"""

import argparse

from classifiers import MyRandomForestClassifier
from data import load_reviews_dataset, split_df
from model_select import GridSearchCV
from vectorizers import MyCountVectorizer, MyWord2Vectorizer


def main(test):
    """
    Main function
    """
    if test:
        parameters = {
            "max_features": ["sqrt"],
            "n_estimators": [500],
            "max_depth": [5],
            "min_samples_split": [5],
            "min_samples_leaf": [1],
            "bootstrap": [True],
        }
    else:
        parameters = {
            "max_features": ["sqrt", "log2"],
            "n_estimators": [500, 1000, 1500],
            "max_depth": [5, 10, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [1, 2, 5, 10],
            "bootstrap": [True, False],
        }
    df = load_reviews_dataset()
    x_train, x_test, y_train, y_test = split_df(df)

    grid_search = GridSearchCV(
        [MyCountVectorizer, MyWord2Vectorizer],
        [MyRandomForestClassifier],
        parameters,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    grid_search.log_best()
    print(grid_search.score(x_test, y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the reviews dataset."
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test set"
    )
    args = parser.parse_args()
    main(args.test)
