from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entities.data_ingestion_config import DataIngestionConfig
from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.entities.artifacts_entity import DataIngestionAftifacts

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
        Read the data from MongoDB and convert it to DataFrame
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            database = self.mongo_client[database_name]
            collections = database[collection_name]

            df = pd.DataFrame(list(collections.find()))

            if "_id" in df.columns.tolist():
                df.drop(columns=["_id"], inplace=True, axis=1)

            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Save full raw dataset into Feature Store inside Artifacts folder.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split dataset into train and test and save into ingested folder.
        """
        try:
            train_df, test_df = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed Train-Test split")

            ingested_dir_path = os.path.dirname(
                self.data_ingestion_config.ingested_train_file_path
            )
            os.makedirs(ingested_dir_path, exist_ok=True)

            logging.info("Exporting Train-Test split data to ingested folder")

            train_df.to_csv(
                self.data_ingestion_config.ingested_train_file_path,
                index=False,
                header=True,
            )
            test_df.to_csv(
                self.data_ingestion_config.ingested_test_file_path,
                index=False,
                header=True,
            )

            logging.info("Successfully exported Train-Test split data")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        """
        Main method to perform data ingestion:
        1. Fetch data from MongoDB
        2. Save in Feature Store
        3. Split into train and test, save in ingested folder
        4. Return artifact with paths
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifacts = DataIngestionAftifacts(
                trained_file_path=self.data_ingestion_config.ingested_train_file_path,
                tested_file_path=self.data_ingestion_config.ingested_test_file_path,
            )

            return data_ingestion_artifacts
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)

    ing = DataIngestion(data_ingestion_config)
    paths = ing.initiate_data_ingestion()

    print("Train file path:", paths.trained_file_path)
    print("Test file path:", paths.tested_file_path)
