import os
'''
defining common constant variable for training pipeline
'''

TARGET_COLUMN: str = 'Result'
RAW_FILE_NAME: str = 'PhishingData.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
ARTIFACT_DIR_NAME: str = 'Artifacts'
PIPLINE_NAME: str = 'NetwordSecurity'

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

MODEL_FILE_NAME = "model.pkl"

SAVED_MODEL_DIR =os.path.join("saved_models")