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
        pass
    else:
        c_vec_params = {
            "min_df": [1],
            "ngram_range": [(1, 1)],
        }
        w2_vec_params = {
            "vector_size": [100],
            "window": [5],
            "min_count": [1],
        }
        model_params = {
            "max_features": ["sqrt", "log2"],
            "n_estimators": [500, 1000, 1500],
            "max_depth": [5, 10, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [1, 2, 5, 10],
            "bootstrap": [True, False],
        }
        save_path = "model.csv"
    df = load_reviews_dataset()
    x_train, x_test, y_train, y_test = split_df(df)
    # x_train, y_train = df["text"], df["label"]

    grid_search = MyGridSearchCV(
        [(MyCountVectorizer, c_vec_params), (MyWord2Vectorizer, w2_vec_params)],
        [(MyRandomForestClassifier, model_params)],
        n_jobs=4,
        log=True,
        save=True,
        save_path=save_path,
    )

    grid_search.fit(x_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the reviews dataset."
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test set"
    )
    args = parser.parse_args()
    main(args.test)
