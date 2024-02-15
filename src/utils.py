import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        logging.info("Model Training Begun")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            logging.info("Checking the best hyperparameters")
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            logging.info("Best hyperparameters found")

            logging.info("Model Training")
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            logging.info("Model training complete")

        

            logging.info("Predicting phase begun")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            logging.info("Predictions Complete")


            logging.info("Noting precision of each model")
            train_model_score = precision_score(y_train, y_train_pred, average='macro')
            test_model_score = precision_score(y_test, y_test_pred, average='macro')


            report[list(models.keys())[i]] = test_model_score
            logging.info("Precision of each model noted")

        return report

    except Exception as e:
        raise CustomException(e, sys)