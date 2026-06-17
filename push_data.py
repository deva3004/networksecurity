import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException

load_dotenv()

username = quote_plus(os.getenv("MONGO_USERNAME", ""))
password = quote_plus(os.getenv("MONGO_PASSWORD", ""))
cluster  = os.getenv("MONGO_CLUSTER", "")

MONGO_DB_URL = (
    f"mongodb+srv://{username}:{password}@{cluster}"
    f"/?retryWrites=true&w=majority"
)

DATABASE_NAME = "NetworkSecurity"
COLLECTION_NAME = "NetworkData"
DATA_PATH = os.path.join("Network_Data", "phisingData.csv")


def load_data(file_path: str) -> list: 
    try:
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        records = json.loads(df.to_json(orient="records"))
        logger.info("Data loaded: %d records from %s", len(records), file_path)
        return records
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def push_to_mongodb(records: list) -> None:
    try:
        client = MongoClient(MONGO_DB_URL, server_api=ServerApi("1"))
        client.admin.command("ping")
        logger.info("Connected to MongoDB successfully")

        collection = client[DATABASE_NAME][COLLECTION_NAME]
        collection.insert_many(records)
        logger.info("Inserted %d records into %s.%s", len(records), DATABASE_NAME, COLLECTION_NAME)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    records = load_data(DATA_PATH)
    push_to_mongodb(records)
    print(f"Successfully pushed {len(records)} records to MongoDB collection '{COLLECTION_NAME}' in database '{DATABASE_NAME}'.")
