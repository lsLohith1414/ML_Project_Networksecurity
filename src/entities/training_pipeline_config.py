from datetime import datetime
from src.constants import training_pipeline
import os
import sys



class TrainingPipelineConfig:
    def __init__(self, timestamp : datetime = None):

        if timestamp is None:
            timestamp = datetime.now()

        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M")

        self.pipline_name = training_pipeline.PIPLINE_NAME
        artifact_name = training_pipeline.ARTIFACT_DIR_NAME
        self.artifact_dir = os.path.join(artifact_name,timestamp)
        self.timestamp = timestamp
        