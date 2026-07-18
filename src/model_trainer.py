import os
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.custom_exeption import CustomExeption
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.inspection import permutation_importance

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self,processed_path,model_output):
        self.procesed_path = processed_path
        self.model_output = model_output
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.scaler = None
        self.clf = None

        os.makedirs(self.model_output,exist_ok=True)
        logger.info("Model training initialized...!")
    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.procesed_path,"X_train.pkl"))
            self.x_test = joblib.load(os.path.join(self.procesed_path,"X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.procesed_path,"y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.procesed_path,"y_test.pkl"))
            self.scaler = joblib.load(os.path.join(self.procesed_path,"scaler.pkl"))

            logger.info("Loaded training,testing data")

        except Exception as e:
            logger.error(f"Faild to load the data {e}")
            raise CustomExeption("Error while model training",e)

    def train(self):
        try:
            self.clf= LogisticRegression(random_state=42,max_iter=1000)
            self.clf.fit(self.x_train,self.y_train) 

            joblib.dump(self.clf,os.path.join(self.model_output,"model.pkl"))
            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Error while training the model...! {e}")
            raise CustomExeption("Error while training the model")

    def evaluate(self):
        try:
            y_pred = self.clf.predict(self.x_test)
            accuracy = accuracy_score(self.y_test,y_pred)
            logger.info(f"Accuracy : {accuracy}")

            precision = precision_score(self.y_test,y_pred,average="weighted")
            logger.info(f"Precision is: {precision}")

            recall = recall_score(self.y_test,y_pred,average="weighted")
            f1score = f1_score(self.y_test,y_pred,average="weighted")
            logger.info(f"Recall: {recall}, F1_score: {f1score}")

            logger.info("Model evluation successful...")

        except Exception as e:
            logger.error(f"Error while evaluation {e}")
            raise CustomExeption("Error while evaluation",e)

    def run(self):
        self.load_data()
        self.train()
        self.evaluate()

if __name__=="__main__":
    trainer = ModelTrainer("artifacts/processed","artifacts/models")
    trainer.run()