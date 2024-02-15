import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            label = ["RiskLevel"]
            features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

            feature_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'median')),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Target label column is {label}")

            logging.info("Preparing the preprocessor object")
            preprocessor = ColumnTransformer(
                [
                    ("feature_pipeline", feature_pipeline, features)
                ]
            )
            logging.info("Preprocessor object prepared")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Taking data from train.csv and test.csv")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Both datasets have been taken")

            logging.info("Getting the preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("The preprocessor object has been loaded")

            target_feature = "RiskLevel"

            logging.info("Making input and target data from the datasets")
            input_train_df = train_df.drop(columns= [target_feature], axis= 1)
            target_train_df = train_df[target_feature]
            input_test_df = test_df.drop(columns= [target_feature], axis= 1)
            target_test_df = test_df[target_feature]
            logging.info("Input and target data taken")


            logging.info("Applying preprocessing on the datasets")
            input_train_array = preprocessor_obj.fit_transform(input_train_df)
            input_test_array = preprocessor_obj.transform(input_test_df)
            logging.info("Preprocessing Complete")


            logging.info("Preparing the training and testing array")
            train_arr = np.c_[input_train_array, np.array(target_train_df)]
            test_arr = np.c_[input_test_array, np.array(target_test_df)]
            logging.info("training and testing array created")


            logging.info("Saving the preprocessor object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info("Preprocessor object saved")


            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            raise CustomException(e,sys)

