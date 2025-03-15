import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from source.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
            
            
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            score = r2_score(y_test, y_pred)
            
            report[list(models.keys())[i]] = score
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    