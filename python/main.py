"""
Main module to load the dataset and train the model.
"""

import argparse

from classifiers import MyRandomForestClassifier
from data import load_reviews_dataset, split_df
from model_select import MyGridSearchCV
from vectorizers import MyCountVectorizer, MyWord2Vectorizer


def main(test):
    """
    Main function
    """
    if test:
        c_vec_params = {
            "min_df": [1, 2],
            "ngram_range": [(1, 1)],
        }
        w2_vec_params = {
            "vector_size": [100],
            "window": [5],
            "min_count": [1, 5],
        }
        model_params = {
            "max_features": ["sqrt"],
            "n_estimators": [500, 1000],
            "max_depth": [5],
            "min_samples_split": [5],
            "min_samples_leaf": [1],
            "bootstrap": [True],
        }
        save_path = "test.csv"
    else:
        c_vec_params = {
            "min_df": [1, 2, 5],
            "ngram_range": [(1, 1), (1, 2), (1, 3)],
        }
        w2_vec_params = {
            "vector_size": [100, 200, 500],
            "window": [5, 10, 20],
            "min_count": [1, 5, 10],
        }
        model_params = {
            "max_features": ["sqrt", "log2"],
            "n_estimators": [500, 1000, 1500],
            "max_depth": [5, 10, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [1, 2, 5, 10],
            "bootstrap": [True, False],
        }
        save_path = "results.csv"
    df = load_reviews_dataset()
    x_train, x_test, y_train, y_test = split_df(df)

    grid_search = MyGridSearchCV(
        [(MyCountVectorizer, c_vec_params), (MyWord2Vectorizer, w2_vec_params)],
        [(MyRandomForestClassifier, model_params)],
        n_jobs=-1,
        log=True,
        save=True,
        save_path=save_path,
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
