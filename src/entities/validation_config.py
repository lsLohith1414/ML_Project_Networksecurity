from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.constants import validation_constants, training_pipeline
import os




class DataValidationConfig:
    def __init__(self,traning_pipeline_config:TrainingPipelineConfig):

        self.data_validation_dir = os.path.join(traning_pipeline_config.artifact_dir, validation_constants.DATA_VALIDATION_DIR_NAME )

        
        self.data_validated_dir = os.path.join(self.data_validation_dir, validation_constants.DATA_VALIDATION_VALID_DIR)

        self.invalid_data_dir = os.path.join(self.data_validation_dir, validation_constants.DATA_VALIDATION_INVALID_DIR)

        self.valid_train_file_path = os.path.join(self.data_validated_dir, training_pipeline.TRAIN_FILE_NAME)

        self.valid_test_file_path = os.path.join(self.data_validated_dir, training_pipeline.TEST_FILE_NAME)

        self.invalid_train_file_path = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)

        self.invalid_test_file_path = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)

        self.drift_report = os.path.join(self.data_validation_dir, validation_constants.DATA_VALIDATION_DRIFT_REPORT_DIR, validation_constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)



        print(self.drift_report)
     



if __name__ == "__main__":
    obj = DataValidationConfig(TrainingPipelineConfig())