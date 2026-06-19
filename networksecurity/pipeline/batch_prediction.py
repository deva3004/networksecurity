import os
import sys
import pandas as pd
from datetime import datetime

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.mains_utils.utils import load_object
from networksecurity.constants.training_pipeline import (
    MODEL_FILE_NAME,
    TARGET_COLUMN,
)

PREDICTION_OUTPUT_DIR = "prediction_output"
FINAL_MODEL_PATH = os.path.join("final_model", MODEL_FILE_NAME)


class BatchPrediction:
    def __init__(self, input_file_path: str, model_path: str = FINAL_MODEL_PATH):
        try:
            self.input_file_path = input_file_path
            self.model_path = model_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _load_input_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_file_path)
            logger.info("Loaded input data: %s rows, %s cols from %s",
                        len(df), len(df.columns), self.input_file_path)

            # Drop target column if accidentally included
            if TARGET_COLUMN in df.columns:
                df.drop(columns=[TARGET_COLUMN], inplace=True)
                logger.info("Dropped target column '%s' from input", TARGET_COLUMN)

            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _get_output_path(self) -> str:
        try:
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
            return os.path.join(PREDICTION_OUTPUT_DIR, f"predictions_{timestamp}.csv")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_batch_prediction(self) -> str:
        try:
            logger.info("========== Batch Prediction Started ==========")

            model = load_object(self.model_path)
            logger.info("Loaded model from: %s", self.model_path)

            df = self._load_input_data()

            predictions = model.predict(df)
            df[TARGET_COLUMN] = predictions

            # Map numeric predictions back to labels (0 → legitimate, 1 → phishing)
            df["prediction_label"] = df[TARGET_COLUMN].map({0: "legitimate", 1: "phishing"})

            output_path = self._get_output_path()
            df.to_csv(output_path, index=False)

            logger.info("Predictions saved to: %s", output_path)
            logger.info(
                "Prediction summary — total: %d | phishing: %d | legitimate: %d",
                len(df),
                int((df[TARGET_COLUMN] == 1).sum()),
                int((df[TARGET_COLUMN] == 0).sum()),
            )
            logger.info("========== Batch Prediction Completed ==========")
            return output_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)
