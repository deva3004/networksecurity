import sys
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger


def main():
    try:
        logger.info("========== Training Pipeline Started ==========")

        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data Ingestion Artifact: %s", data_ingestion_artifact)

        logger.info("========== Training Pipeline Completed ==========")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    main()
