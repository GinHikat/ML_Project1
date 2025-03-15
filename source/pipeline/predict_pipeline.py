import sys
import pandas as pd
from source.exception import CustomException
from source.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            prep_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            prep = load_object(file_path = prep_path)
            data_scaled = prep.transform(features)
            pred = model.predict(data_scaled)
            
            return pred
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education: str,
                 lunch: int,
                 test_preparation_course:int,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data(self):
        try:
            data_input = {
                'gender': [self.gender],
                'race_ethnicity':[self.race_ethnicity if self.race_ethnicity is not None else "Group A"],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch':[self.lunch if self.lunch is not None else "Unknown"],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }
            
            return pd.DataFrame(data_input)
        
        except Exception as e:
            raise CustomException(e, sys)
        