import sys
from networksecurity.logging import logger


def get_error_message(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script: [{file_name}] at line [{line_number}] — {str(error)}"


class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = get_error_message(error_message, error_detail)
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
