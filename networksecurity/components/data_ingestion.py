import os
import sys
import certifi
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

    def export_collection_as_dataframe(self) -> pd.DataFrame:

        '''Export MongoDB collection data as a pandas DataFrame'''
         
        try:
            username = quote_plus(os.getenv("MONGO_USERNAME", ""))
            password = quote_plus(os.getenv("MONGO_PASSWORD", ""))
            cluster  = os.getenv("MONGO_CLUSTER", "")
            mongo_url = (
                f"mongodb+srv://{username}:{password}@{cluster}"
                f"/?retryWrites=true&w=majority"
            )

            client = MongoClient(mongo_url, tlsCAFile=certifi.where())
            collection = client[DATA_INGESTION_DATABASE_NAME][
                self.data_ingestion_config.collection_name
            ]

            df = pd.DataFrame(list(collection.find()))
            logger.info("Exported %d records from MongoDB collection '%s'",
                        len(df), self.data_ingestion_config.collection_name)

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace("na", pd.NA, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        '''Save the DataFrame to the feature store directory'''

        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info("Data saved to feature store: %s", feature_store_file_path)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:

        '''Split the DataFrame into training and testing sets and save them to disk'''

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True
            )

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logger.info("Train set saved: %s", self.data_ingestion_config.training_file_path)
            logger.info("Test set saved: %s", self.data_ingestion_config.testing_file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Initiating data ingestion")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info("Data ingestion artifact: %s", data_ingestion_artifact)
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
