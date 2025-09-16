import sys

def get_custom_exception(error,error_details:sys):
    exc_type,exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"[Error occur in python script name [{file_name}] in the line [{exc_tb.tb_lineno}] and error message is [{str(error)}] ]"
    return error_message


class NetworkSecurityException(Exception):
    def __init__(self, error,error_details:sys):
        super().__init__(error)
        self.error_message = get_custom_exception(error,error_details)

    def __str__(self):
        return self.error_message
    


# if __name__ == "__main__":
#     try:
#         a = 8/0
#     except Exception as e:
#         raise CustomException(e,sys)    