import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "raw_data.csv")
    sampled_data_path : str = os.path.join("artifacts", "sampled_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process")

        try:
            df = pd.read_csv('notebook/data/maternal_health_risk_new.csv')
            logging.info("File has been read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            logging.info("Loading data into the raw_data.csv")
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Loaded data into the raw_data.csv")

            logging.info("Undersampling the dominant class")
            sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            df, df["RiskLevel"] = sampler.fit_resample(df, df["RiskLevel"])
            df.to_csv(self.ingestion_config.sampled_data_path, index= False, header=True)
            logging.info("Dominant undersampling completed")

            logging.info("Initiating train/test split")
            train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
            logging.info("Train/test split completed")

            logging.info("Loading the train and test datasets to their respective csv files")
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Loading the train and test datasets complete")

            logging.info("Data Ingestion is complete")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            return CustomException(e, sys)



if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()



