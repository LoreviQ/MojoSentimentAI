import ast

import matplotlib.pyplot as plt
import pandas as pd


def read_results(path):
    df = pd.read_csv(
        path,
        header=None,
    )
    # Concatenate the columns that were split incorrectly
    df[3] = (
        df[3].astype(str)
        + ","
        + df[4].astype(str)
        + ","
        + df[5].astype(str)
        + ","
        + df[6].astype(str)
        + ","
        + df[7].astype(str)
        + ","
        + df[8].astype(str)
    )
    df[4] = df[9].astype(str) + "," + df[10].astype(str) + "," + df[11].astype(str)
    df = df.drop(columns=[5, 6, 7, 8, 9, 10, 11])
    df.columns = ["Score", "Model", "Vectorizer", "Model Params", "Vectorizer Params"]
    return df


def plot_vectorizer(df):
    df = df.drop(columns=["Model", "Model Params"])
    vectorizer_groups = {
        vectorizer: sub_df for vectorizer, sub_df in df.groupby("Vectorizer")
    }
    for vectorizer, sub_df in vectorizer_groups.items():
        # Extract parameters from Vectorizer Params
        params = sub_df["Vectorizer Params"].apply(ast.literal_eval)

        # Get all unique parameter keys
        param_keys = set()
        for param in params:
            param_keys.update(param.keys())

        # Plot each parameter
        for key in param_keys:
            sub_df[key] = params.apply(lambda x: x.get(key, None))
            if key == "ngram_range":
                sub_df[key] = sub_df[key].apply(lambda x: x[1])

            # Encode other parameters into a single value for color
            color_values = params.apply(
                lambda x: hash(tuple(sorted((k, v) for k, v in x.items() if k != key)))
            )

            plt.figure()
            scatter = plt.scatter(
                sub_df[key], sub_df["Score"], c=color_values, cmap="viridis", marker="x"
            )
            plt.title(f"Vectorizer: {vectorizer} - Parameter: {key}")
            plt.xlabel(key)
            plt.ylabel("Score")
            plt.colorbar(scatter, label="Other Params")
            plt.grid(True)
            plt.show()


def plot_model(df):
    df = df.drop(columns=["Vectorizer", "Vectorizer Params"])
    model_groups = {model: sub_df for model, sub_df in df.groupby("Model")}
    for model, sub_df in model_groups.items():
        # Extract parameters from Model Params
        params = sub_df["Model Params"].apply(ast.literal_eval)

        # Get all unique parameter keys
        param_keys = set()
        for param in params:
            param_keys.update(param.keys())

        # Plot each parameter
        for key in param_keys:
            sub_df[key] = params.apply(lambda x: x.get(key, None))

            # Encode other parameters into a single value for color
            color_values = params.apply(
                lambda x: hash(tuple(sorted((k, v) for k, v in x.items() if k != key)))
            )

            plt.figure()
            scatter = plt.scatter(
                sub_df[key], sub_df["Score"], c=color_values, cmap="viridis", marker="x"
            )
            plt.title(f"Model: {model} - Parameter: {key}")
            plt.xlabel(key)
            plt.ylabel("Score")
            plt.colorbar(scatter, label="Other Params")
            plt.grid(True)
            plt.show()


def plot():
    df = read_results("./../results/emoticons_model_params.csv")
    plot_model(df)


if __name__ == "__main__":
    plot()
