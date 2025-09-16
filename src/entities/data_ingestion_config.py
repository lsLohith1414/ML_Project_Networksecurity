from datetime import datetime
import os 
from src.constants import training_pipeline, data_ingestion
from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.logging.logger import logging





class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, data_ingestion.DATA_INGESTION_DIR_NAME)

        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, data_ingestion.DATA_INGESTION_FEATURE_STORE_NAME, training_pipeline.RAW_FILE_NAME)

        self.ingested_train_file_path: str = os.path.join(self.data_ingestion_dir, data_ingestion.DATA_INGESTION_INGESTED_NAME, training_pipeline.TRAIN_FILE_NAME)

        self.ingested_test_file_path: str = os.path.join(self.data_ingestion_dir, data_ingestion.DATA_INGESTION_INGESTED_NAME, training_pipeline.TEST_FILE_NAME)

        self.database_name: str = data_ingestion.DATA_INGESTION_DATABASE_NAME

        self.collection_name: str = data_ingestion.DATA_INGESTION_COLLECTION_NAME

        self.train_test_split_ratio: float = data_ingestion.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO



if __name__ == "__main__":
    training_pipline_config = TrainingPipelineConfig()

    ingestion_config = DataIngestionConfig(training_pipline_config)


    print(ingestion_config.feature_store_file_path)
    logging.info("printing the file paths raw file paths")
    print(ingestion_config.ingested_train_file_path)
    logging.info("printing the file paths train file paths")
    print(ingestion_config.ingested_test_file_path)
    logging.info("printing the file paths test file paths")

    

