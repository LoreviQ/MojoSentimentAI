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
    path = "./../results/amazon_reviews_sklearn_test.csv"
    df = pd.read_csv(path)
    columns = [
        "param_bootstrap",
        "param_max_depth",
        "param_max_features",
        "param_min_samples_leaf",
        "param_min_samples_split",
        "param_n_estimators",
        "mean_test_score",
        "std_test_score",
    ]
    df = df.loc[:, columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, param in zip(axes.flatten(), columns[:-2]):
        # Individual formatting for each parameter
        if param == "param_bootstrap":
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["False", "True"])
        elif param == "param_max_depth":
            ax.set_xticks([0] + sorted(df["param_max_depth"].unique()))
            ax.set_xticklabels(
                ["None"]
                + [str(x) for x in sorted(df["param_max_depth"].unique()) if x != 0]
            )
            df[param] = df[param].fillna(0)

        # General formatting
        param_label = param.replace("param_", "").replace("_", " ").title()
        avg_scores = df.groupby(param)["mean_test_score"].mean()
        min_scores = df.groupby(param)["mean_test_score"].min()
        max_scores = df.groupby(param)["mean_test_score"].max()
        avg_stds = df.groupby(param)["std_test_score"].mean()
        ax.plot(
            avg_scores.index,
            avg_scores.values,
            color="green",
            label="Mean Test Score",
        )
        ax.scatter(
            min_scores.index,
            min_scores.values,
            color="blue",
            marker="x",
            s=100,
            label="Min/Max Scores",
        )
        ax.scatter(
            max_scores.index,
            max_scores.values,
            color="blue",
            marker="x",
            s=100,
        )
        ax.errorbar(
            avg_scores.index,
            avg_scores.values,
            yerr=avg_stds.values,
            fmt="o",
            color="black",
            ecolor="gray",
            elinewidth=2,
            capsize=4,
            label="Std Dev",
        )
        ax.set_title(f"{param_label} vs Mean Test Score")
        ax.set_xlabel(param_label)
        ax.set_ylabel("Test Score")
        ax.set_ylim(0, 1)

        # Other formatting
        ax.axhline(y=0.5, color="red", linestyle="--")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
