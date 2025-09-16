from src.constants import transfromation_constants, training_pipeline
from src.entities.training_pipeline_config import TrainingPipelineConfig
import os


class TransformationConfig:
    def __init__(self, trining_pipeline_config:TrainingPipelineConfig):

        self.transformation_dir = os.path.join(trining_pipeline_config.artifact_dir, transfromation_constants.DATA_TRANSFORMATION_DIR_NAME)

        self.transformed_train_file_path: str = os.path.join( self.transformation_dir,transfromation_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.transformation_dir,  transfromation_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        
        self.transformed_object_file_path: str = os.path.join( self.transformation_dir, transfromation_constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)
        

        print(self.transformation_dir)
        print(self.transformed_train_file_path)
        print(self.transformed_test_file_path)
        print(self.transformed_object_file_path)
        



if __name__ == "__main__":
    obj = TransformationConfig(TrainingPipelineConfig())
