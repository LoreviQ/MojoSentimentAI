import time

import numpy as np
import pandas as pd
from data import load_amazon_reviews_dataset
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def train_model(train_df, ratio, c_vec_params, model_params):
    train_df = train_df.sample(frac=ratio)
    c_vec = CountVectorizer(**c_vec_params)
    x_train = c_vec.fit_transform(train_df["text"])
    y_train = train_df["label"]
    model = RandomForestClassifier(**model_params)
    model.fit(x_train, y_train)
    return c_vec, model


def process_ratio_cv(ratio, cv, df, c_vec_params, model_params):
    test_df = df.sample(frac=0.2)
    train_df = df.drop(test_df.index)
    start_time = time.time()
    vectorizer, model = train_model(train_df, ratio, c_vec_params, model_params)
    training_time = time.time() - start_time
    x_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]
    score = model.score(x_test, y_test)
    print(f"Ratio: {ratio} - CV: {cv} - Score: {score} - Took: {training_time:.2f}s")
    return [ratio, cv, score, training_time]


def model_collapse_sklearn():
    df = load_amazon_reviews_dataset()
    df["text"] = df["text"].apply(lambda x: " ".join(x))

    c_vec_params = {
        "min_df": 1,
        "ngram_range": (1, 1),
    }
    model_params = {
        "max_features": "sqrt",
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": False,
    }
    results = Parallel(n_jobs=-1)(
        delayed(process_ratio_cv)(ratio, cv, df, c_vec_params, model_params)
        for ratio in get_ratios(30)
        for cv in range(10)
    )
    results_df = pd.DataFrame(results, columns=["ratio", "cv", "score", "time"])
    results_df.to_csv(
        "./../results/amazon_reviews_sklearn_collapse_log.csv", index=False
    )


def get_ratios(n, ratio=1):
    if n == 0:
        return []
    return get_ratios(n - 1, ratio * 0.75) + [ratio]


if __name__ == "__main__":
    model_collapse_sklearn()
