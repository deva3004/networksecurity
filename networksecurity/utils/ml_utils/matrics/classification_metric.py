import sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException


def get_classification_score(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
