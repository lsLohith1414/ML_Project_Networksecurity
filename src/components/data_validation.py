from src.entities.artifacts_entity import DataIngestionAftifacts, DataValidationArtifact
from src.entities.validation_config import DataValidationConfig
from src.exception.exception import NetworkSecurityException
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from src.entities.training_pipeline_config import TrainingPipelineConfig
from src.logging.logger import logging
from src.utilites.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import os
import sys
import pandas as pd


class DataValidation:
    def __init__(self, data_ingestion_artifacts: DataIngestionAftifacts,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logging.info(f"Loaded schema: {self._schema_config}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_num_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if number of columns in dataframe matches schema.
        """
        try:
            number_of_columns = len(self._schema_config["columns"])  # schema.yaml should have "columns"
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")

            return len(dataframe.columns) == number_of_columns

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """
        Perform KS test to detect data drift between train and test.
        """
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    drift_found = False
                else:
                    drift_found = True
                    status = False

                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": drift_found
                    }
                })

            drift_report_file_path = self.data_validation_config.drift_report
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifacts.trained_file_path
            test_file_path = self.data_ingestion_artifacts.tested_file_path

            # Read train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            error_message = ""

            # Validate train
            train_status = self.validate_num_of_columns(train_df)
            if not train_status:
                error_message += "Train dataframe does not contain all required columns.\n"

            # Validate test
            test_status = self.validate_num_of_columns(test_df)
            if not test_status:
                error_message += "Test dataframe does not contain all required columns.\n"

            overall_status = train_status and test_status
            if not overall_status:
                raise Exception(error_message)

            # Check data drift
            drift_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save validated datasets
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_ingestion_artifacts.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifacts.tested_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report,
            )

            logging.info(f"Data validation completed: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    # ⚠️ Replace below with proper artifact paths when you integrate
    dummy_ingestion_artifact = DataIngestionAftifacts(
        trained_file_path="Artifacts/train.csv",
        tested_file_path="Artifacts/test.csv"
    )
    data_validation_config = DataValidationConfig(training_pipeline_config)
    obj = DataValidation(dummy_ingestion_artifact, data_validation_config)
    obj.initiate_data_validation()
