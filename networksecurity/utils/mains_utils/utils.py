import os
import sys
import yaml
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)
        logger.info("YAML file written: %s", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
        logger.info("Numpy array saved: %s", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Object saved: %s", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    param_grids: dict,
) -> dict:
    """
    Run GridSearchCV for each model in `models` using its corresponding param grid.

    Returns a report dict keyed by model name:
        {
            "best_estimator": <fitted estimator>,
            "best_params":    {param: value, ...},
            "cv_f1":          float,   # mean CV score on train
            "train_f1":       float,
            "test_f1":        float,
        }
    """
    try:
        report = {}

        for name, model in models.items():
            logger.info("Running GridSearchCV for %s ...", name)

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring="f1",
                cv=3,
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            train_f1 = f1_score(y_train, best_estimator.predict(X_train))
            test_f1  = f1_score(y_test,  best_estimator.predict(X_test))

            report[name] = {
                "best_estimator": best_estimator,
                "best_params":    grid_search.best_params_,
                "cv_f1":          grid_search.best_score_,
                "train_f1":       train_f1,
                "test_f1":        test_f1,
            }

            logger.info(
                "[%s] best_params=%s | CV F1=%.4f | Train F1=%.4f | Test F1=%.4f",
                name, grid_search.best_params_,
                grid_search.best_score_, train_f1, test_f1,
            )

        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)