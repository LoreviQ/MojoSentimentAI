import time
from multiprocessing import Manager

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

vec_params_base = {
    "min_df": 1,
    "ngram_range": (1, 1),
}
model_params_base = {
    "max_features": "sqrt",
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": False,
}
save_path = "./../results/generational_collapse.csv"
df = pd.read_csv("./../datasets/amazon_reviews_tokens.csv")


class GenerationalModel:
    def __init__(
        self,
        lock,
        generation=0,
        collapse_rate=0.05,
        vec_params=None,
        model_params=None,
        previous_vectorizer=None,
        previous_model=None,
        cv=5,
    ):
        start_time = time.time()
        self.lock = lock
        self.generation = generation
        self.collapse_rate = collapse_rate
        if self.generation > 0:
            self.collapse_rate += np.random.normal() * 0.01
        self.collapse_rate = max(0, min(1, self.collapse_rate))
        self.vec_params = vec_params
        self.model_params = model_params
        self.previous_vectorizer = previous_vectorizer
        self.previous_model = previous_model
        self.cv = cv
        self.vectorizer = self._randomise_vectorizer()
        self.model = self._randomise_model()
        self.score = self._train_test()
        self.time_taken = time.time() - start_time
        self.log_result()

    def _randomise_vectorizer(self):
        if self.vec_params is None:
            self.vec_params = vec_params_base
        # randomise min_df
        self.vec_params["min_df"] += int(round(np.random.normal()))
        self.vec_params["min_df"] = max(1, self.vec_params["min_df"])
        # randomise ngram_range
        ngram_val = self.vec_params["ngram_range"][1] + int(round(np.random.normal()))
        ngram_val = max(1, ngram_val)
        self.vec_params["ngram_range"] = (1, ngram_val)
        return CountVectorizer(**self.vec_params)

    def _randomise_model(self):
        if self.model_params is None:
            self.model_params = model_params_base
        # randomise max_features
        if np.random.rand() > 0.95:
            if self.model_params["max_features"] == "sqrt":
                self.model_params["max_features"] = "log2"
            else:
                self.model_params["max_features"] = "sqrt"
        # randomise n_estimators
        self.model_params["n_estimators"] += int(round(np.random.normal() * 100))
        self.model_params["n_estimators"] = max(1, self.model_params["n_estimators"])
        # randomise max_depth
        self.model_params["max_depth"] += int(round(np.random.normal() * 5))
        self.model_params["max_depth"] = max(1, self.model_params["max_depth"])
        # randomise min_samples_split
        self.model_params["min_samples_split"] += int(round(np.random.normal()))
        self.model_params["min_samples_split"] = max(
            2, self.model_params["min_samples_split"]
        )
        # randomise min_samples_leaf
        self.model_params["min_samples_leaf"] += int(round(np.random.normal()))
        self.model_params["min_samples_leaf"] = max(
            1, self.model_params["min_samples_leaf"]
        )
        # randomise bootstrap
        if np.random.rand() > 0.9:
            self.model_params["bootstrap"] = not self.model_params["bootstrap"]
        return RandomForestClassifier(**self.model_params)

    def _train_test(self):
        scores = []
        for _ in range(self.cv):
            train_df = df.sample(n=10000)
            test_df = df.drop(train_df.index).sample(n=10000)
            train_df = self._collapse_df(train_df)
            x_train = self.vectorizer.fit_transform(train_df["text"])
            y_train = train_df["label"]
            self.model.fit(x_train, y_train)
            x_test = self.vectorizer.transform(test_df["text"])
            y_test = test_df["label"]
            scores.append(self.model.score(x_test, y_test))
        return np.mean(scores)

    def _collapse_df(self, train_df):
        if self.generation == 0:
            return train_df
        if self.previous_model is None or self.previous_vectorizer is None:
            raise ValueError("Previous model and vectorizer must be set")
        selected_rows = train_df.sample(frac=self.collapse_rate).index
        if not selected_rows.empty:
            train_df.loc[selected_rows, "label"] = self.previous_model.predict(
                self.previous_vectorizer.transform(train_df.loc[selected_rows, "text"])
            )
        return train_df

    def create_next_generation(self, n):
        next_gen = Parallel(n_jobs=-1)(
            delayed(GenerationalModel)(
                lock=self.lock,
                generation=self.generation + 1,
                collapse_rate=self.collapse_rate,
                vec_params=self.vec_params,
                model_params=self.model_params,
                previous_vectorizer=self.vectorizer,
                previous_model=self.model,
            )
            for _ in range(n)
        )
        return next_gen

    def log_result(self, save=True, log=True):
        # log for printing progress
        # save for saving to file
        if log:
            print(
                f"Generation: {self.generation} - Score: {self.score:.2f} - Collapse Rate: {self.collapse_rate:.2f} - Took: {self.time_taken:.2f}s"
            )
        if save:
            vec_vals = ",".join([f'"{v}"' for v in self.vec_params.values()])
            model_vals = ",".join([f"{v}" for v in self.model_params.values()])
            with self.lock:
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{self.generation},{self.score},{self.collapse_rate},{vec_vals},{model_vals},{self.time_taken}\n"
                    )


def create_next_generation_global(best_model, generational_n):
    return best_model.create_next_generation(generational_n)


def train_generational_model(initial_n, generational_n, num_best):
    with Manager() as manager:
        lock = manager.Lock()
        best_models = []
        generation = 0
        # Train initial models
        print(f"Training generation {generation}")
        models = Parallel(n_jobs=-1)(
            delayed(GenerationalModel)(lock) for _ in range(initial_n)
        )
        end_training = False
        while True:
            end_training = True
            # Get best models
            new_best_models = 0
            for model in models:
                if len(best_models) < num_best:
                    best_models.append(model)
                    end_training = False
                    new_best_models += 1
                else:
                    for i, best_model in enumerate(best_models):
                        if model.score > best_model.score:
                            best_models[i] = model
                            end_training = False
                            new_best_models += 1
                            break
            # End training if no new best models
            if end_training:
                break
            # Create next generation
            generation += 1
            print(f"Found {new_best_models} new best models")
            print(f"Training generation {generation}")
            models = Parallel(n_jobs=-1)(
                delayed(create_next_generation_global)(best_model, generational_n)
                for best_model in best_models
            )
            models = [model for sublist in models for model in sublist]

        print("Completed training")
        print("Best models:")
        for best_model in best_models:
            best_model.log_result(save=False)


if __name__ == "__main__":
    train_generational_model(100, 10, 10)
