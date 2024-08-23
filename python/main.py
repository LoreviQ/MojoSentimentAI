"""
Main module to load the dataset and train the model.
"""

import argparse

from data import load_reviews_dataset
from textVectorizers import CustomCountVectorizer, Word2Vectorizer
from training import train_model, train_test_split


def main(test):
    """
    Main function
    """
    if test:
        df = load_reviews_dataset()
        x_train, x_test, y_train, y_test = train_test_split(df, Word2Vectorizer())
        print(x_train.shape)

    else:
        df = load_reviews_dataset()
        print("--- CountVectorizer ---")
        x_train, x_test, y_train, y_test = train_test_split(df, CustomCountVectorizer())
        model = train_model(x_train, y_train)
        print("--- Score: " + str(model.score(x_test, y_test)) + " ---")
        print("--- Word2Vectorizer ---")
        x_train, x_test, y_train, y_test = train_test_split(df, Word2Vectorizer())
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
