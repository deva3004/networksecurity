import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            config = DataIngestionConfig(self.training_pipeline_config)
            artifact = DataIngestion(config).initiate_data_ingestion()
            logger.info("Data ingestion complete: %s", artifact)
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")
            config = DataValidationConfig(self.training_pipeline_config)
            artifact = DataValidation(data_ingestion_artifact, config).initiate_data_validation()
            logger.info("Data validation complete: %s", artifact)
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")
            config = DataTransformationConfig(self.training_pipeline_config)
            artifact = DataTransformation(data_validation_artifact, config).initiate_data_transformation()
            logger.info("Data transformation complete: %s", artifact)
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")
            config = ModelTrainerConfig(self.training_pipeline_config)
            artifact = ModelTrainer(data_transformation_artifact, config).initiate_model_trainer()
            logger.info("Model training complete: %s", artifact)
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            logger.info("========== Training Pipeline Started ==========")

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            logger.info("========== Training Pipeline Completed ==========")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
