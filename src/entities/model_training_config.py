from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.constants import model_training_constants
import os


class ModelTrainerConfig:
    def __init__(self, training_pipline_config: TrainingPipelineConfig):
        self.model_trainer_dir:str = os.path.join(training_pipline_config.artifact_dir, model_training_constants.MODEL_TRAINER_DIR_NAME )

        self.trained_model_file_path = os.path.join(self.model_trainer_dir, model_training_constants.MODEL_TRAINER_TRAINED_MODEL_DIR,model_training_constants.MODEL_TRAINER_TRAINED_MODEL_NAME)

        self.excepted_accuracy = model_training_constants.MODEL_TRAINER_EXPECTED_SCORE

        self.overfitting_underfitting_threshold = model_training_constants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD