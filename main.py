"""
This module is a python template of my intended mojo ai build

Functions:
    load_dataset(): Load the dataset from a CSV file and clean it.
    clean_dataset(df): Clean the dataset by dropping unnecessary columns and rows with missing values.
    main(): Main function to load and print the dataset.
"""

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

nltk.download("stopwords")
nltk.download("wordnet")


def load_dataset():
    """
    Load the dataset from a CSV file and clean it.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = pd.read_csv("reviews_dataset.csv")
    df = clean_dataset(df)
    return df


def clean_dataset(df):
    """
    Clean the dataset by dropping unnecessary columns and rows with missing values.

    Args:
        df (pd.DataFrame): The dataset to clean.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"sentence": "text"})
    df = df.dropna()
    df["text"] = df["text"].apply(process_text)
    return df


def process_text(text):
    """
    Process the text data by removing special characters and converting it to lowercase.

    Args:
        text (str): The text data to process.

    Returns:
        str: The processed text data.
    """
    lm = WordNetLemmatizer()

    text = text.lower()  # Convert to lowercase
    text = "".join(
        [char for char in text if char.isalnum() or char.isspace()]
    )  # Remove special characters
    words = text.split()  # Tokenize the text
    text = " ".join(
        [lm.lemmatize(word) for word in words if word not in stopwords.words("english")]
    )  # Lemmatize the text and remove stopwords
    return text


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


def main():
    """
    Main function to load and print the dataset.
    """
    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df)
    model = train_model(x_train, y_train)
    print(model.score(x_test, y_test))


if __name__ == "__main__":
    main()
