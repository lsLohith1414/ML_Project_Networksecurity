import os 
import sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
# entities: -> configuration files
from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.entities.data_ingestion_config import DataIngestionConfig 
from src.entities.validation_config import DataValidationConfig  
from src.entities.transformation_config import TransformationConfig  
from src.entities.model_training_config import ModelTrainerConfig   

# components: -> main working files:
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Artifacts: -> output of the main working fiels
from src.entities.artifacts_entity import(
    DataIngestionAftifacts,
    DataValidationArtifact,
    DataTransformationArtifact,
    ClassificationMetricArtifact,
    ModelTrainerArtifact
)


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipline_config = TrainingPipelineConfig()
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def start_data_ingestion(self):
        try:
            
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipline_config)
            logging.info("Start data Ingestion")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

            logging.info(f"Successfully completed Data Ingestion and Artfacts: {data_ingestion_artifacts}")

            return data_ingestion_artifacts
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)        

    def start_data_validation(self, data_ingestion_artifacts:DataIngestionAftifacts):
        try:
            
            self.data_validation_config = DataValidationConfig(traning_pipeline_config=self.training_pipline_config)
            logging.info("Start Data Validation")

            data_validation = DataValidation(data_ingestion_artifacts=data_ingestion_artifacts, data_validation_config= self.data_validation_config)

            data_validation_artifacts = data_validation.initiate_data_validation() 

            logging.info(f"Successfully completed Data Validation and Artifacts: {data_validation_artifacts}")

            return data_validation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def start_data_transformation(self, data_validation_artifacts : DataValidationArtifact):
        try:
            self.data_transformation_config = TransformationConfig(trining_pipeline_config=self.training_pipline_config)
            logging.info("Start Data Transformation")

            data_transformation = DataTransformation(data_validation_artifacts=data_validation_artifacts, transformation_config=self.data_transformation_config)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info(f"Successfully completed Data Transformation and Artifacts: {data_transformation_artifacts}")

            return data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    
    def start_model_trainer(self, data_transformation_artifacts:DataTransformationArtifact)->ModelTrainerArtifact:

        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipline_config=self.training_pipline_config)
            logging.info("Start Model Training")

            model_trainer = ModelTrainer(model_trainer_config= self.model_trainer_config, data_transformation_artifacts=data_transformation_artifacts)

            model_trainer_artifacts = model_trainer.initiate_model_triner()

            logging.info(f"Successfully completed Model Training and Artifacts: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def run_training_pipline(self):

        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts=data_ingestion_artifacts)

            data_transformation_artifacts = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)

            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts=data_transformation_artifacts)

            return model_trainer_artifacts


        except Exception as e:
            raise NetworkSecurityException(e,sys)









