"""
Contains search algorithms for hyperparameter tuning.
"""

from itertools import product
from multiprocessing import Manager

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class GridSearchCV:
    """
    Custom implementation of the GridSearchCV class.
    """

    def __init__(
        self,
        vectorizers,
        models,
        cv=5,
        return_train_score=False,
        n_jobs=4,
        scoring="accuracy",
        log=True,
        save=False,
        save_path=None,
    ):
        self.vectorizers = vectorizers
        self.models = models
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.best_vectorizer_ = None
        self.best_vectorizer_instance_ = None
        self.best_model_instance_ = None
        self.log = log
        self.save = save
        self.save_path = save_path
        self.results = []

    def cross_validate(
        self, model, vectorizer, model_params, vectorizer_params, x, y, lock
    ):
        """
        Perform cross-validation on the model.
        """
        kf = KFold(n_splits=self.cv)
        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_fold_validation)(
                model,
                vectorizer,
                model_params,
                vectorizer_params,
                x,
                y,
                train_index,
                val_index,
                i,
            )
            for i, (train_index, val_index) in enumerate(kf.split(x), 1)
        )
        result = [
            np.mean(scores),
            model,
            vectorizer,
            model_params,
            vectorizer_params,
        ]
        if self.save:
            self.save_result(result, lock)
        else:
            self.results.append(result)

    def _single_fold_validation(
        self,
        model,
        vectorizer,
        model_params,
        vectorizer_params,
        x,
        y,
        train_index,
        val_index,
        fold_index,
    ):
        """
        Perform a single fold validation.
        """
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        vectorizer_instance = vectorizer(**vectorizer_params)
        x_train_vec = vectorizer_instance.fit_transform(x_train)
        x_val_vec = vectorizer_instance.transform(x_val)

        model_instance = model(**model_params)
        model_instance.fit(x_train_vec, y_train)
        y_pred = model_instance.predict(x_val_vec)

        if self.scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError("Unsupported scoring method")

        if self.log:
            self.log_progress(
                vectorizer, model, vectorizer_params, model_params, fold_index
            )

        return score

    def fit(self, x, y):
        """
        Fit the model on the data.
        """
        with Manager() as manager:
            lock = manager.Lock()
            Parallel(n_jobs=self.n_jobs)(
                delayed(self.cross_validate)(
                    model[0],
                    vectorizer[0],
                    dict(zip(list(model[1].keys()), model_param_combination)),
                    dict(zip(list(vectorizer[1].keys()), vectorizer_param_combination)),
                    x,
                    y,
                    lock,
                )
                for model in self.models
                for vectorizer in self.vectorizers
                for model_param_combination in product(*list(model[1].values()))
                for vectorizer_param_combination in product(
                    *list(vectorizer[1].values())
                )
            )
        if self.save is False:
            # Handle results
            results_array = np.array(self.results, dtype=object)
            max_index = np.argmax(results_array[:, 0])
            (
                best_score,
                best_model,
                best_vectorizer,
                best_model_params,
                best_vectorizer_params,
            ) = results_array[max_index]
            self.best_score_ = best_score
            self.best_model_ = (best_model, best_model_params)
            self.best_vectorizer_ = (best_vectorizer, best_vectorizer_params)
            self.best_vectorizer_instance_ = best_vectorizer(**best_vectorizer_params)
            self.best_vectorizer_instance_.fit(x)
            self.best_model_instance_ = best_model(**best_model_params)
            self.best_model_instance_.fit(
                self.best_vectorizer_instance_.transform(x), y
            )

        print("\n --- DONE ---")

    def predict(self, x):
        """
        Make predictions using the best model.
        """
        x_vec = self.best_vectorizer_instance_.transform(x)
        return self.best_model_instance_.predict(x_vec)

    def score(self, x, y):
        """
        Calculate the accuracy of the model.
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_best_model(self):
        """
        Get the best model instance
        """
        return self.best_model_instance_

    def get_best_vectorizer(self):
        """
        Get the best vectorizer instance
        """
        return self.best_vectorizer_instance_

    def get_best_params(self):
        """
        Get the best parameters
        """
        return self.best_model_[1]

    def get_best_score(self):
        """
        Get the best score
        """
        return self.best_score_

    def log_progress(self, vectorizer, model, vectorizer_params, model_params, i):
        """
        Log the progress of the hyperparameter tuning.
        """
        model_name = model.__name__
        vectorizer_name = vectorizer.__name__

        message = f"{vectorizer_name} - {vectorizer_params} - {model_name} - {model_params} - {i}/{self.cv} ---"
        print(message, end="\r")

    def log_best(self):
        """
        Log the best hyperparameters and model.
        """
        print(f"Best score: {self.best_score_}")
        print(f"Best model: {self.best_model_[0].__name__}")
        print(f"Best vectorizer: {self.best_vectorizer_[0].__name__}")
        print(f"Best model parameters: {self.best_model_[1]}")
        print(f"Best vectorizer parameters: {self.best_vectorizer_[1]}")

    def save_result(self, result, lock):
        """
        Save a result to a CSV file.
        """

        if self.save is False:
            return
        if self.save_path is None:
            raise ValueError("No save path provided")
        with lock:
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{result[0]},{result[1].__name__},{result[2].__name__},{result[3]},{result[4]}\n"
                )
