from src.components.data_ingestion import DataIngestion
from src.entities.data_ingestion_config import DataIngestionConfig
from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entities.validation_config import DataValidationConfig 
from src.entities.artifacts_entity import DataIngestionAftifacts,DataValidationArtifact, DataTransformationArtifact
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entities.transformation_config import TransformationConfig
import sys

from src.components.model_trainer import ModelTrainer
from src.entities.model_training_config import ModelTrainerConfig

if __name__ == "__main__":
    try:
        # Step 1: Create pipeline-level config
        training_pipeline_config = TrainingPipelineConfig()

        # Step 2: Create data ingestion config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        
        # Step 3: Run Data Ingestion
        ingestion = DataIngestion(data_ingestion_config)
        paths = ingestion.initiate_data_ingestion()

        # Step 4: Print the artifact outputs
        print("✅ Train file saved at:", paths.trained_file_path)
        print("✅ Test file saved at:", paths.tested_file_path)

        logging.info("Data ingestion Completed")

        # Step 5 logging.info("Initiate Data ingestion")
        data_ingestion_artifacts = DataIngestionAftifacts(paths.trained_file_path,paths.tested_file_path)
        data_validation_config = DataValidationConfig(TrainingPipelineConfig())

        data_validation = DataValidation(data_ingestion_artifacts,data_validation_config)
        
        logging.info("Initiate Data Validation")
        data_validation_artifacts = data_validation.initiate_data_validation()
        # print(data_validation)
        logging.info("Completed Data Validation")



        # Validation artifact is passing to data Transformation 

        logging.info("Data Transformation started")

        transformation_config = TransformationConfig(training_pipeline_config)

        data_transformation = DataTransformation(data_validation_artifacts , transformation_config )

        data_transformation_artifacts = data_transformation.initiate_data_transformation()

        print(data_transformation_artifacts)

        logging.info("Data Transformation completed")


        # model training 
        model_train_config = ModelTrainerConfig(training_pipeline_config)

        model_trainer = ModelTrainer(model_trainer_config=model_train_config, data_transformation_artifacts= data_transformation_artifacts)

        model_train_artifacts = model_trainer.initiate_model_triner()

        logging.info("successfully completed model training")
        print(model_train_artifacts)





    except Exception as e:
        raise NetworkSecurityException(e, sys)
