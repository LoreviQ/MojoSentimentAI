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
    df = pd.read_csv("./../datasets/reviews_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"sentence": "text"})
    df = df.dropna()
    df["text"] = df["text"].apply(tokenize_text)
    return df
