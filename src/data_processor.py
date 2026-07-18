import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.custom_exeption import CustomExeption
from src.logger import get_logger
import os

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None

        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data processing started..!")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Loaded the data")
        except Exception as e:
            logger.error(f"Error while loading the data {e}")
            raise CustomExeption("Error while data loading",e)

    def process(self):
        try:
            self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"],errors="coerce")
            cat_columns = ["Operation_Mode","Efficiency_Status"]
            for col in cat_columns:
                self.df[col] = self.df[col].astype('category')

            self.df["year"] = self.df["Timestamp"].dt.year
            self.df["month"] = self.df["Timestamp"].dt.month
            self.df["day"] = self.df["Timestamp"].dt.day

            self.df["hour"]= self.df["Timestamp"].dt.hour

            columns_to_encode = ["Efficiency_Status","Operation_Mode"]
            le = LabelEncoder()
            for col in columns_to_encode:
                self.df[col] = le.fit_transform(self.df[col])
            logger.info('All the preprocessing data')
            
        except Exception as e:
            logger.error("Error while data processing")
            raise CustomExeption("Error while data processing",e)

    def split_and_scale_save(self):
        try:
            self.features = [
                'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
                'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
                'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
                'Predictive_Maintenance_Score', 'Error_Rate_%','year', 'month', 'day', 'hour'
            ]

            X = self.df[self.features]
            y=self.df["Efficiency_Status"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train , X_test , y_train , y_test = train_test_split(X_scaled,y, test_size=0.2 , random_state=42 , stratify=y)

            joblib.dump(X_train,os.path.join(self.output_path,"X_train.pkl"))
            joblib.dump(X_test,os.path.join(self.output_path,"X_test.pkl"))
            joblib.dump(y_train,os.path.join(self.output_path,"y_train.pkl"))
            joblib.dump(y_test,os.path.join(self.output_path,"y_test.pkl"))

            joblib.dump(scaler,os.path.join(self.output_path,"scaler.pkl"))

            logger.info("All things saved successfully...!")

        except Exception as e:
            logger.error(f"Error while data spliting scale and save{e}")
            raise CustomExeption("Error while data spliting and save",e)

    def run(self):
        self.load_data()
        self.process()
        self.split_and_scale_save()

if __name__=="__main__":
    processor = DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    processor.run()