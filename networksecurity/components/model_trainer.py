import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.mains_utils.utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    evaluate_models,
)
from networksecurity.utils.ml_utils.matrics.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.models.estimater import NetworkModel

CANDIDATE_MODELS = {
    "RandomForest":       RandomForestClassifier(random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(random_state=42),
    "AdaBoost":           AdaBoostClassifier(random_state=42),
    "DecisionTree":       DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
}

PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators":      [64, 128, 256],
        "max_depth":         [None, 8, 16],
        "min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "n_estimators":  [64, 128],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [3, 5],
    },
    "AdaBoost": {
        "n_estimators":  [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
    "DecisionTree": {
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion":         ["gini", "entropy"],
    },
    "LogisticRegression": {
        "C":      [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    },
}


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(
        self,
        model_name: str,
        model,
        best_params: dict,
        train_metrics: ClassificationMetricArtifact,
        test_metrics: ClassificationMetricArtifact,
    ) -> None:
        try:
            mlflow.set_experiment("NetworkSecurity")

            with mlflow.start_run(run_name=model_name):
                mlflow.set_tag("model_name", model_name)

                # Hyperparameters chosen by GridSearchCV
                mlflow.log_params(best_params)

                # Train metrics
                mlflow.log_metric("train_f1",        train_metrics.f1_score)
                mlflow.log_metric("train_precision",  train_metrics.precision_score)
                mlflow.log_metric("train_recall",     train_metrics.recall_score)

                # Test metrics
                mlflow.log_metric("test_f1",         test_metrics.f1_score)
                mlflow.log_metric("test_precision",   test_metrics.precision_score)
                mlflow.log_metric("test_recall",      test_metrics.recall_score)

                # Log the raw sklearn estimator (not the NetworkModel wrapper)
                mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info("MLflow run logged for model: %s", model_name)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Initiating model training with hyperparameter tuning")

            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr  = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:,  :-1], test_arr[:,  -1]

            report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=CANDIDATE_MODELS,
                param_grids=PARAM_GRIDS,
            )

            best_name  = max(report, key=lambda name: report[name]["test_f1"])
            best_info  = report[best_name]
            best_model = best_info["best_estimator"]

            logger.info("Best model: %s  (Test F1: %.4f)", best_name, best_info["test_f1"])

            train_metrics = get_classification_score(y_train, best_model.predict(X_train))
            test_metrics  = get_classification_score(y_test,  best_model.predict(X_test))

            logger.info(
                "[%s] Train — F1: %.4f  Precision: %.4f  Recall: %.4f",
                best_name, train_metrics.f1_score,
                train_metrics.precision_score, train_metrics.recall_score,
            )
            logger.info(
                "[%s] Test  — F1: %.4f  Precision: %.4f  Recall: %.4f",
                best_name, test_metrics.f1_score,
                test_metrics.precision_score, test_metrics.recall_score,
            )

            # Track this run in MLflow
            self.track_mlflow(
                model_name=best_name,
                model=best_model,
                best_params=best_info["best_params"],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )

            if test_metrics.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model ({best_name}) F1 {test_metrics.f1_score:.4f} is below "
                    f"expected threshold {self.model_trainer_config.expected_accuracy}"
                )

            gap = abs(train_metrics.f1_score - test_metrics.f1_score)
            if gap > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(
                    f"Best model ({best_name}) overfitting detected — "
                    f"train/test F1 gap {gap:.4f} exceeds threshold "
                    f"{self.model_trainer_config.overfitting_underfitting_threshold}"
                )

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            save_object(self.model_trainer_config.trained_model_file_path, network_model)
            logger.info("Saved %s: %s", best_name, self.model_trainer_config.trained_model_file_path)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
