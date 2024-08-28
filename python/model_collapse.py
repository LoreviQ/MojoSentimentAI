import time

import pandas as pd
from data import load_amazon_reviews_dataset
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def score_model(test_df, vectorizer, model):
    x_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]
    return model.score(x_test, y_test)


def train_model(train_df, c_vec_params, model_params):
    start_time = time.time()
    c_vec = CountVectorizer(**c_vec_params)
    x_train = c_vec.fit_transform(train_df["text"])
    y_train = train_df["label"]
    model = RandomForestClassifier(**model_params)
    model.fit(x_train, y_train)
    training_time = time.time() - start_time
    return c_vec, model, training_time


def process_ratio_cv(collapse_ratio, n, cv, df, c_vec_params, model_params):
    result = []
    test_df = df.sample(frac=0.2)
    train_df = df.drop(test_df.index)
    initial_ratio = get_ratio(n)
    collapse_df, model, vectorizer = (None, None, None)
    while n >= 0:
        size_ratio = get_ratio(n)
        collapse_df = expand_collapse_df(
            size_ratio, collapse_ratio, train_df, model, vectorizer, collapse_df
        )
        vectorizer, model, training_time = train_model(
            collapse_df, c_vec_params, model_params
        )
        score = score_model(test_df, vectorizer, model)
        result += [[initial_ratio, size_ratio, cv, score, training_time]]
        n = n - 1
        print(
            f"Initial Ratio: {initial_ratio:.2g} - Ratio: {size_ratio:.2g} - Collapse Ratio: {collapse_ratio} - CV: {cv} - Score: {score} - Took: {training_time:.2f}s"
        )
    return result


def expand_collapse_df(
    size_ratio, collapse_ratio, train_df, model=None, vectorizer=None, collapse_df=None
):
    if collapse_df is None:
        return train_df.sample(frac=size_ratio)

    new_rows = train_df.sample(frac=size_ratio).shape[0] - collapse_df.shape[0]
    expansion_df = train_df.sample(n=new_rows)
    selected_rows = expansion_df.sample(frac=collapse_ratio).index
    expansion_df.loc[selected_rows, "label"] = model.predict(
        vectorizer.transform(expansion_df.loc[selected_rows, "text"])
    )
    collapse_df = pd.concat([collapse_df, expansion_df])
    return collapse_df


def model_collapse_sklearn():
    df = load_amazon_reviews_dataset()
    df["text"] = df["text"].str.join(" ")

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
        delayed(process_ratio_cv)(1, n, cv, df, c_vec_params, model_params)
        for n in range(10)
        for cv in range(5)
    )
    results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(
        results, columns=["initial_ratio", "new_ratio", "cv", "score", "time"]
    )
    results_df.to_csv(
        "./../results/amazon_reviews_sklearn_collapse_log_cr1.csv", index=False
    )


def get_ratio(n):
    return 0.39810717055349725**n


if __name__ == "__main__":
    model_collapse_sklearn()
