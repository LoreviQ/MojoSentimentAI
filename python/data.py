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
    df["text"] = df["text"].apply(process_text)
    return df
