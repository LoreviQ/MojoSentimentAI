"""
Main module to load the dataset and train the model.
"""

from data import load_reviews_dataset
from training import train_model, train_test_split


def main():
    """
    Main function
    """
    df = load_reviews_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df)
    model = train_model(x_train, y_train)
    print(model.score(x_test, y_test))


if __name__ == "__main__":
    main()
