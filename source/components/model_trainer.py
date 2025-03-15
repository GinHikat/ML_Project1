import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from source.exception import CustomException
from source.logger import logging

from source.utils import save_object, evaluate_model

@dataclass
class ModelTrainConfig:
    train_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()
        
    def init_model_train(self, train, test):
        try:
            logging.info('Splitting train and test')
            
            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1]    
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=15, n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(criterion='squared_error', max_depth=5, min_samples_split=2, splitter='best'),
                "Random Forest Regressor": RandomForestRegressor(max_depth=None, max_features=8, min_samples_split=8, n_estimators=400),
                "XGBRegressor": XGBRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor(n_estimators=400),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'Gradient Boosting': GradientBoostingRegressor()
            }     
            
            model_result:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, 
                                         models = models)
            
            #Show best score and the corresponding model
            best_score = max(sorted(model_result.values()))
            
            best_model_name = list(model_result.keys())[
                list(model_result.values()).index(best_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best model found')
            
            save_object(
                file_path=self.model_train_config.train_model_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2 = r2_score(y_test, predicted)
            return r2
            
            

        except Exception as e:
            raise CustomException(e, sys)