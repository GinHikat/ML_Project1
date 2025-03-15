import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from source.exception import CustomException
from source.logger import logging

from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_trans = DataTransformationConfig()
        
    def get_trans(self):
        
        '''
        This function transforms data
        '''
        
        try:
            num_col = ['writing_score', 'reading_score']
            cat_col = ['gender',
                       'race_ethnicity',
                       'parental_level_of_education',
                       'lunch',
                       'test_preparation_course']
            
            num_pipe = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy= 'median')),
                    ('Scaler', StandardScaler())
                ]
            )
            
            cat_pipe = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('Encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info('Numerical columns encoded')
            logging.info('Categorical columns encoded')
            
            preprocesser = ColumnTransformer(
                [
                    ('num_pipeline', num_pipe, num_col),
                    ('cat_pipeline', cat_pipe, cat_col)
                ]
            )
            
            return preprocesser
        except Exception as e:
            raise CustomException(e, sys)
        
    def init_data_transform(self, train_path, test_path):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            
            
            logging.info('Load train and test from csv')
            
            prep = self.get_trans()
            
            target = 'math_score'
            num_col = ['writing_score', 'reading_score']
            
            input_train = train.drop(columns = [target], axis = 1)
            target_train = train[target]
            
            input_test = test.drop(columns = [target], axis = 1)
            target_test = test[target]
            
            logging.info(
               f"Apply transformation for training and testing" 
            )
    
            prep.fit(input_train)
            input_train = prep.fit_transform(input_train)
            input_test = prep.transform(input_test)
            
            train_arr = np.c_[input_train, np.array(target_train)]
            test_arr = np.c_[input_test, np.array(target_test)]
            
            logging.info(f'Save prep object')
            
            save_object(
                file_path = self.data_trans.preprocessor_path,
                obj = prep
            )
            
            return (
                train_arr,
                test_arr,
                self.data_trans.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)