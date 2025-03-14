import sys
from source.logger import logging

def error_message(error, detail:sys):
    _,_,exc = detail.exc_info() #which file, which line the exception occurs
    file_name = exc.tb_frame.f_code.co_filename
    error_message = f'Error occured in {0} line {1} with message {2}'
    file_name, exc.tb_lineno, str(error)
    
    return error_message
    
class CustomException(Exception):
    def __init__(self, message, detail:sys):
        super().__init__(message)
        self.message = error_message(error_message, detail = detail)
        
    def __str__(self):
        return self.message