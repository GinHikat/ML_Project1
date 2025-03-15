import os
import sys
from source.exception import CustomException
from source.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from source.components.data_transformation import DataTransformation, DataTransformationConfig

from source.components.model_trainer import ModelTrainConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info('Initialized data ingestion')
        try:
            df = pd.read_csv('main_data/data/stud.csv')
            logging.info('Read data from csv')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info('Train test split')
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            
            train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
        
            logging.info('Data ingestion completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train, test = obj.init_data_ingestion()
    
    data_trans = DataTransformation()
    train, test, _ = data_trans.init_data_transform(train, test)
    
    modeltrain = ModelTrainer()
    print(modeltrain.init_model_train(train, test))