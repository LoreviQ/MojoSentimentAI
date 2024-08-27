"""
This module is used to import different data sources and return them as pandas dataframes.
The exported dataframe will always have the following columns:
- text: The text data.
- label: The label data.
"""

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


def tokenize_text(text):
    """
    Process the text data by tokenizing, converting to lowercase,
    removing special characters, lemmatizing, and removing stopwords

    Args:
        text (str): The text data to process.

    Returns:
        list: The processed tokens.
    """
    lm = WordNetLemmatizer()

    text = text.lower()
    text = "".join([char for char in text if char.isalnum() or char.isspace()])
    words = text.split()
    tokens = [
        lm.lemmatize(word) for word in words if word not in stopwords.words("english")
    ]
    return tokens


def load_reviews_dataset():
    """
    Load the reviews dataset from a CSV file and clean it.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = pd.read_csv("./../datasets/reviews.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"sentence": "text"})
    df = df.dropna()
    df["text"] = df["text"].apply(tokenize_text)
    return df


def load_emoticons_dataset():
    """
    Load the emoticons dataset from a CSV file and clean it.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = pd.read_csv("./../datasets/emoticons.csv", header=None)
    df.columns = ["text", "label"]
    df = df.dropna()
    df["text"] = df["text"].apply(tokenize_text)
    return df


def load_amazon_reviews_dataset(rows=None):
    """
    Load the Amazon reviews dataset from a CSV file and clean it.

    Args:
        rows (int, optional): Number of rows to read from the CSV file. If None, read the entire dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = pd.read_csv("./../datasets/amazon_reviews.csv", nrows=rows)
    df.columns = ["text", "label"]
    df = df.dropna()
    df["text"] = df["text"].apply(tokenize_text)
    return df


def split_df(df, ratio=0.2):
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
    x_train = train["text"].reset_index(drop=True)
    y_train = train["label"].reset_index(drop=True)
    x_test = test["text"].reset_index(drop=True)
    y_test = test["label"].reset_index(drop=True)
    return x_train, x_test, y_train, y_test
