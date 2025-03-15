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
                "Decision Tree": DecisionTreeRegressor(criterion='squared_error', max_depth=5, min_samples_split=2, splitter='best'),
                "Random Forest Regressor": RandomForestRegressor(max_depth=None, max_features=8, min_samples_split=8, n_estimators=400),
                "XGBRegressor": XGBRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor(n_estimators=400),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'Gradient Boosting': GradientBoostingRegressor()
            }     
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_result:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, 
                                         models = models, params =params)
            
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