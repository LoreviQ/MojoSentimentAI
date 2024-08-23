"""
Main module to load the dataset and train the model.
"""

import argparse

from data import load_reviews_dataset
from textVectorizers import CustomCountVectorizer
from training import train_model, train_test_split


def main(test):
    """
    Main function
    """
    if test:
        df = load_reviews_dataset()
        cv = CustomCountVectorizer(ngram_range=(1, 4))
        cv.fit_transform(df["text"])

    else:
        df = load_reviews_dataset()
        print("--- CountVectorizer ngram_range=(1, 1) ---")
        x_train, x_test, y_train, y_test = train_test_split(df, CustomCountVectorizer())
        model = train_model(x_train, y_train)
        print("--- Score: " + str(model.score(x_test, y_test)) + " ---")
        print("--- CountVectorizer ngram_range=(1, 2) ---")
        x_train, x_test, y_train, y_test = train_test_split(
            df, CustomCountVectorizer(ngram_range=(1, 2))
        )
        model = train_model(x_train, y_train)
        print("--- Score: " + str(model.score(x_test, y_test)) + " ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the reviews dataset."
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test set"
    )
    args = parser.parse_args()
    main(args.test)
