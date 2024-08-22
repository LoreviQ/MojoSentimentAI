"""
This module is a python template of my intended mojo ai build

Functions:
    load_dataset(): Load the dataset from a CSV file and clean it.
    clean_dataset(df): Clean the dataset by dropping unnecessary columns and rows with missing values.
    main(): Main function to load and print the dataset.
"""

import pandas as pd


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
    df = df.dropna()
    return df


def main():
    """
    Main function to load and print the dataset.
    """
    df = load_dataset()
    print(df)


if __name__ == "__main__":
    main()
