import sys
import logging

def error_message_details(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including the error message and the file name where the error occurred.
    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module to access the traceback.
    Returns:
        str: A formatted string containing the error message and the file name.
    """
    # Extract traceback details (exception information)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Construct the error message
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"

    # Log the error message for better tracking
    logging.error(error_message)
    return error_message

class MyException(Exception):
    """
    Custom exception class that inherits from the base Exception class.
    This can be used to raise specific exceptions in the application.
    """
    def __init__(self, error_message: str, error_detail: sys = sys):
        """
        Initializes the custom exception with a detailed error message.
        Args:
            error_message (str): The error message to be included in the exception.
            error_detail (sys): The sys module to access traceback details.
        """
        # Call the base class constructor
        super().__init__(error_message)
        # Store the error message with detailed information
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """
        Returns a string representation of the exception.
        Returns:
            str: The error message stored in the exception.
        """
        return self.error_message