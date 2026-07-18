from src.data_processor import DataProcessing
from src.model_trainer import ModelTrainer


if __name__=="__main__":
    processor = DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    processor.run()

    trainer = ModelTrainer("artifacts/processed","artifacts/models")
    trainer.run()