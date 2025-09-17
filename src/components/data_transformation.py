import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.constants.training_pipeline import TARGET_COLUMN
from src.constants.transfromation_constants import DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.entities.transformation_config import TransformationConfig
from src.entities.artifacts_entity import DataTransformationArtifact, DataValidationArtifact
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utilites.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifacts: DataValidationArtifact, transformation_config:TransformationConfig):
        
        try:
            self.data_validation_artifacts = data_validation_artifacts
            self.transformation_config = transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(cls)->Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        """
        logging.info(
            "Entered get_data_trnasformer_object method of Trnasformation class"
        )


        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            processor:Pipeline = Pipeline([("imputer",imputer)])

            return processor


        except Exception as e:
            raise NetworkSecurityException(e,sys)




        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation calss")
        
        try:
            logging.info("Statrting Data Transformation")

            train_df = DataTransformation.read_data(self.data_validation_artifacts.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifacts.valid_test_file_path)

            

            # Train dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            # Test dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            preprocessor = self.get_data_transformer_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_features = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)] 

            # save numpy array data 

            save_numpy_array_data(file_path=self.transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.transformation_config.transformed_test_file_path, array=test_arr)

            save_object(file_path=self.transformation_config.transformed_object_file_path, obj=preprocessor_obj)

            save_object("final_models/preprocessor.pkl",preprocessor)

            # preparing the artifacts 
            data_transformation_artifacts = DataTransformationArtifact(
                transformed_object_file_path= self.transformation_config.transformed_object_file_path,
                transformed_train_file_path= self.transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.transformation_config.transformed_test_file_path
            )

            return data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e,sys)

        