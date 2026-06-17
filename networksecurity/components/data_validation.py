import os
import sys
import yaml
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH, DATA_VALIDATION_THRESHOLD


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema = self._read_schema()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _read_schema(self) -> dict:
        try:
            with open(SCHEMA_FILE_PATH, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected = len(self._schema["columns"])
            actual = len(dataframe.columns)
            if actual != expected:
                logger.warning("Column count mismatch — expected %d, got %d", expected, actual)
                return False
            logger.info("Column count validated: %d columns", actual)
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected = set(self._schema["columns"].keys())
            actual = set(dataframe.columns)
            missing = expected - actual
            extra = actual - expected
            if missing or extra:
                logger.warning("Missing columns: %s | Extra columns: %s", missing, extra)
                return False
            logger.info("All column names validated successfully")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            drift_found = False
            report = {}

            for col in base_df.columns:
                stat, p_value = ks_2samp(base_df[col].dropna(), current_df[col].dropna())
                drifted = p_value < DATA_VALIDATION_THRESHOLD
                report[col] = {
                    "p_value": float(p_value),
                    "ks_statistic": float(stat),
                    "drift_status": drifted,
                }
                if drifted:
                    drift_found = True

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            with open(self.data_validation_config.drift_report_file_path, "w") as f:
                yaml.dump(report, f, default_flow_style=False)

            drifted_cols = [c for c, v in report.items() if v["drift_status"]]
            logger.info("Drift report saved. Drifted columns (%d): %s", len(drifted_cols), drifted_cols)
            return not drift_found
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Initiating data validation")

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Validate columns on train set
            if not self.validate_number_of_columns(train_df):
                raise Exception("Train data failed column count validation")
            if not self.validate_column_names(train_df):
                raise Exception("Train data failed column name validation")

            # Validate columns on test set
            if not self.validate_number_of_columns(test_df):
                raise Exception("Test data failed column count validation")
            if not self.validate_column_names(test_df):
                raise Exception("Test data failed column name validation")

            # Detect drift (train as base, test as current)
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Save valid data
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logger.info("Data validation artifact: %s", data_validation_artifact)
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
