import sys
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException


class NetworkModel:
    """Bundles a fitted preprocessor and a fitted estimator for single-object inference."""

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            transformed = self.preprocessor.transform(X)
            return self.model.predict(transformed)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
