from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class GridSearchCV:
    def __init__(
        self,
        vectorizers,
        models,
        model_params,
        cv=5,
        return_train_score=False,
        n_jobs=4,
        scoring="accuracy",
    ):
        self.vectorizers = vectorizers
        self.models = models
        self.model_params = model_params
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.best_vectorizer_ = None
        self.best_vectorizer_instance_ = None
        self.best_model_instance_ = None
        self.progress = 0
        self.length = 0

    def cross_validate(self, model, vectorizer, params, x, y):
        kf = KFold(n_splits=self.cv)
        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_fold_validation)(
                model, vectorizer, params, x, y, train_index, val_index, i
            )
            for i, (train_index, val_index) in enumerate(kf.split(x), 1)
        )
        return np.mean(scores)

    def _single_fold_validation(
        self, model, vectorizer, params, x, y, train_index, val_index, fold_index
    ):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        vectorizer_instance = vectorizer()
        x_train_vec = vectorizer_instance.fit_transform(x_train)
        x_val_vec = vectorizer_instance.transform(x_val)

        model_instance = model(**params)
        model_instance.fit(x_train_vec, y_train)
        y_pred = model_instance.predict(x_val_vec)

        if self.scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError("Unsupported scoring method")

        # Log the progress
        self.log_progress(score, model, vectorizer, params, fold_index)

        return score

    def fit(self, x, y):
        self.length = (
            len(self.models)
            * len(self.vectorizers)
            * len(list(product(*list(self.model_params.values()))))
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.cross_validate)(
                model,
                vectorizer,
                dict(zip(list(self.model_params.keys()), param_combination)),
                x,
                y,
            )
            for model in self.models
            for vectorizer in self.vectorizers
            for param_combination in product(*list(self.model_params.values()))
        )

        # Find the best result
        for i, (model, vectorizer, param_combination) in enumerate(
            product(
                self.models,
                self.vectorizers,
                product(*list(self.model_params.values())),
            )
        ):
            if results[i] > self.best_score_:
                self.best_score_ = results[i]
                self.best_params_ = dict(
                    zip(list(self.model_params.keys()), param_combination)
                )
                self.best_model_ = model
                self.best_vectorizer_ = vectorizer
                self.best_vectorizer_instance_ = vectorizer()
                self.best_vectorizer_instance_.fit(x)
                self.best_model_instance_ = model(**self.best_params_)
                self.best_model_instance_.fit(
                    self.best_vectorizer_instance_.transform(x), y
                )

        print("\n --- DONE ---")

    def predict(self, x):
        x_vec = self.best_vectorizer_instance_.transform(x)
        return self.best_model_instance_.predict(x_vec)

    def score(self, x, y):
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy

    def best_model(self):
        return self.best_model_instance_

    def best_vectorizer(self):
        return self.best_vectorizer_instance_

    def best_params(self):
        return self.best_params_

    def best_score(self):
        return self.best_score_

    def log_progress(self, score, model, vectorizer, param_combination, i):
        model_name = model.__name__
        vectorizer_name = vectorizer.__name__

        message = f"SCORE: {score} - {model_name} - {vectorizer_name} - {param_combination} - {i}/{self.cv} ---"
        print(message, end="\r")

    def log_best(self):
        print(f"Best score: {self.best_score_}")
        print(f"Best model: {self.best_model_.__name__}")
        print(f"Best vectorizer: {self.best_vectorizer_.__name__}")
        print(f"Best parameters: {self.best_params_}")
