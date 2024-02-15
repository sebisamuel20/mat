import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from xgboost.sklearn import XGBClassifier


from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Arrays have been successfully split")


            models = {
                "Logistic Regression": LogisticRegression(class_weight= 'balanced'),
                "Naive Bayes": GaussianNB(),
                "K nearest Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(class_weight= 'balanced'),
                "Random Forest Classifier": RandomForestClassifier(class_weight= 'balanced'),
                "Gradient Boosting Classifier": GradientBoostingClassifier()
            }

            params = {
                "Logistic Regression": {
                    #  'C':  np.logspace(-4, 4, 50),
                    #  'penalty' : ['l1', 'l2']
                },
                "Naive Bayes": {},
                "K nearest Classifier": {
                    # 'n_neighbors': [1,2,5,7,10,15,21]
                },
                "Decision Tree Classifier": {
                    # 'max_depth': [2,3,5,7,11,13],
                    # 'min_samples_split': [2,3,5],
                    # 'max_features': [None, 'sqrt', 'log2']
                },
                "Random Forest Classifier": {
                    # 'n_estimators': [50,100,150,200,250],
                    # 'max_depth': [None, 1, 4, 7, 10, 20],
                    # 'min_samples_split': [2,3,5,6],
                    # 'max_features': [None, 'auto', 'sqrt', 'log2']
                },
                "Gradient Boosting Classifier": {
                    # 'learning_rate':  [.01,.05,.10,.001],
                    # 'n_estimators':[100, 200, 256, 300, 500],
                    # 'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8], 
                    # 'criterion': ["friedman_mse", "squared_error"],
                    # 'max_depth': [1,3,5,9,15],
                    # 'max_features': [None, 'sqrt', 'log2', 'auto']
                }
            }


            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            
            best_model_score = max(sorted(model_report.values()))

            

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            precision = precision_score(y_test, predicted, average='macro')
            return precision


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    data_train, data_test = obj.initiate_data_ingestion()

    dat = DataTransformation()
    train_arr, test_arr, preproc_obj = dat.initiate_data_transformation(data_train, data_test)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))