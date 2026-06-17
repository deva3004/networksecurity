import sys
from networksecurity.logging.logger import logger


def get_error_message(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info() ##Get the exception information from the sys module
    file_name = exc_tb.tb_frame.f_code.co_filename ##₹Get the file name where the exception occurred
    line_number = exc_tb.tb_lineno ##Get the line number where the exception occurred 
    return f"Error in script: [{file_name}] at line [{line_number}] — {str(error)}" ##Return a formatted error message with the file name, line number, and error message


class NetworkSecurityException(Exception): ##Custom exception class for network security errors"""
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message)) 
        self.error_message = get_error_message(error_message, error_detail) ##Get the formatted error message using the get_error_message function
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
