import os
import sys
from src.entities.model_training_config import ModelTrainerConfig
from src.entities.artifacts_entity import DataTransformationArtifact   # input
from src.entities.artifacts_entity import ModelTrainerArtifact        # output
from src.utilites.ml_utils.metrics.classification_metrics import get_classification_score
from src.utilites.main_utils.utils import (
    load_numpy_array_data, load_object, save_object
)
from src.utilites.ml_utils.model.estimetor import NetworkModel
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import mlflow


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Run GridSearchCV for each model, train with best parameters, 
    and return model scores + trained models.
    """
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            para = params[model_name]

            logging.info(f"Running GridSearchCV for {model_name}...")
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_   # already trained with best params

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Classification accuracy (you can switch to F1, Precision, Recall etc.)
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            logging.info(f"{model_name} -> Train Acc: {train_score}, Test Acc: {test_score}")

            report[model_name] = test_score
            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise NetworkSecurityException(e, sys)


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        


    def track_mlflow(self, model, classification_metrics):
        try:
            f1 = classification_metrics.f1_score
            precision = classification_metrics.precision_score
            recall = classification_metrics.recall_score

            # Example MLflow logging
            import mlflow
            with mlflow.start_run():
                mlflow.log_param("model", type(model).__name__)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("recall_score", recall)

        except Exception as e:
            raise NetworkSecurityException(e, sys)








    def train_model(self, X_train, y_train, X_test, y_test):
        logging.info("Starting training with multiple models...")

        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            # "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            # "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                # 'max_features': ['sqrt', 'log2'],
            },
            "Random Forest": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            # "Gradient Boosting": {
            #     'loss': ['log_loss', 'exponential'],
            #     'learning_rate': [.1, .01, .05, .001],
            #     'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
            #     'criterion': ['squared_error', 'friedman_mse'],
            #     'max_features': ['sqrt', 'log2'],
            #     'n_estimators': [8, 16, 32, 64, 128, 256]
            # },
            "Logistic Regression": {},
            # "AdaBoost": {
            #     'learning_rate': [.1, .01, .001],
            #     'n_estimators': [8, 16, 32, 64, 128, 256]
            # }
        }
        # Run evaluation
        model_report, trained_models = evaluate_models(
            X_train, y_train, X_test, y_test, models, params
        )

        # Pick best model
        best_model_name = max(model_report, key=model_report.get)
        best_model_score = model_report[best_model_name]
        best_model = trained_models[best_model_name]

        logging.info(f"Best model selected: {best_model_name} with Test Accuracy: {best_model_score}")

        # Train metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metrics = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )

         # track ml flow
        self.track_mlflow(best_model,classification_train_metrics)

        # Test metrics
        y_test_pred = best_model.predict(X_test)
        classification_test_metrics = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )



        # track ml flow
        self.track_mlflow(best_model,classification_test_metrics)




        # Load preprocessor
        preprocessor = load_object(
            file_path=self.data_transformation_artifacts.transformed_object_file_path
        )

        # Save model as pickle
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        # Return artifacts
        model_trainer_artifacts = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metrics,
            test_metric_artifact=classification_test_metrics
        )

        logging.info(f"Model trainer artifacts: {model_trainer_artifacts}")

        return model_trainer_artifacts

    def initiate_model_triner(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model training...")

            train_file_path = self.data_transformation_artifacts.transformed_train_file_path
            test_file_path = self.data_transformation_artifacts.transformed_test_file_path

            # Load arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifacts = self.train_model(X_train, y_train, X_test, y_test)

            logging.info("Model training completed successfully")
            return model_trainer_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)
