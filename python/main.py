"""
Main module to load the dataset and train the model.
"""

import argparse

import pandas as pd
from classifiers import MyRandomForestClassifier
from data import load_amazon_reviews_dataset, split_df
from model_select import MyGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from vectorizers import MyCountVectorizer


def sklearn_version():
    """
    Train a model on the Amazon reviews dataset using sklearn.
    Sanity checking the implementation to compare with the custom implementation.
    """
    # Load dataset
    df = load_amazon_reviews_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2
    )

    # Ensure df["text"] contains raw text strings
    if isinstance(df["text"].iloc[0], list):
        df["text"] = df["text"].str.join(" ")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2
    )

    # Vectorize x
    c_vec_params = {
        "min_df": 1,
        "ngram_range": (1, 1),
    }
    c_vec = CountVectorizer(**c_vec_params)
    x_train = c_vec.fit_transform(x_train)
    x_test = c_vec.transform(x_test)

    # Train model
    model_params = {
        "max_features": ["sqrt", "log2"],
        "n_estimators": [100, 500, 1000, 2000, 4000],
        "max_depth": [1, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 5, 10, 20, 40],
        "bootstrap": [True, False],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        model_params,
        cv=5,
        n_jobs=-1,
        return_train_score=True,
        scoring="accuracy",
        verbose=10,
    )

    # Fit and evaluate model
    grid_search.fit(x_train, y_train)
    print("Best Params: " + str(grid_search.best_params_))
    print("Best Score: " + str(grid_search.best_score_))
    print("Test Score: " + str(grid_search.score(x_test, y_test)))

    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv("./../results/amazon_reviews_sklearn_test.csv", index=False)


def my_version():
    """
    Train a model on the Amazon reviews dataset using the custom implementation.
    """

    # load dataset
    df = load_amazon_reviews_dataset(rows=1000)
    x_train, x_test, y_train, y_test = split_df(df)

    # Train model
    c_vec_params = {
        "min_df": [1],
        "ngram_range": [(1, 1)],
    }
    model_params = {
        "max_features": ["sqrt", "log2"],
        "n_estimators": [500, 1000, 1500],
        "max_depth": [5, 10, None],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2, 5, 10],
        "bootstrap": [True, False],
    }
    grid_search = MyGridSearchCV(
        [(MyCountVectorizer, c_vec_params)],
        [(MyRandomForestClassifier, model_params)],
        n_jobs=4,
        log=True,
        save=True,
        save_path="./../results/emoticons_model_params.csv",
        memory=True,
    )

    # Fit and evaluate model
    grid_search.fit(x_train, y_train)
    grid_search.log_best()
    grid_search.score(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the reviews dataset."
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test set"
    )
    args = parser.parse_args()
    if args.test:
        sklearn_version()
    else:
        my_version()
