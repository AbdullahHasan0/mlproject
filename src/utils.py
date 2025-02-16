import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path,obj):
    '''
    This function will save a Python object into a pickle file
    '''
    try:
        dir_path = os.path.dirname(file_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)