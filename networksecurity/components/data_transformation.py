import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.mains_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            return Pipeline(steps=[("imputer", imputer)])
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _load_and_prepare(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            # Replace -1 labels with 0 (phishing datasets use -1/1; models expect 0/1)
            df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(-1, 0)
            features = df.drop(columns=[TARGET_COLUMN])
            target = df[TARGET_COLUMN]
            return features, target
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Initiating data transformation")

            train_features, train_target = self._load_and_prepare(
                self.data_validation_artifact.valid_train_file_path
            )
            test_features, test_target = self._load_and_prepare(
                self.data_validation_artifact.valid_test_file_path
            )

            preprocessor = self.get_data_transformer_object()
            train_features_transformed = preprocessor.fit_transform(train_features)
            test_features_transformed = preprocessor.transform(test_features)

            train_arr = np.c_[train_features_transformed, np.array(train_target)]
            test_arr = np.c_[test_features_transformed, np.array(test_target)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            save_object("final_model/preprocessor.pkl", preprocessor)

            logger.info(
                "Data transformation complete — train shape: %s, test shape: %s",
                train_arr.shape, test_arr.shape,
            )

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
