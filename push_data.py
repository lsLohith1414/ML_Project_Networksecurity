import os
import sys
import json
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URL from .env file
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print("MongoDB URL loaded successfully.")

# Load CA certificate
ca = certifi.where()
print("CA certificate loaded:", ca)

# CustomException (You can define it elsewhere in your project)
from src.exception.exception import CustomException
from src.logging.logger import logging


class NetworkDataExtract:
    def __init__(self):
        try:
            pass  # You can initialize other resources here if needed
        except Exception as e:
            raise CustomException(e, sys)

    def cv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            record = list(json.loads(data.T.to_json()).values())
            return record
        except Exception as e:
            raise CustomException(e, sys)

    def insert_data_mongodb(self, record, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tls=True, tlsCAFile=ca)
            db = self.mongo_client[database]
            collection_obj = db[collection]
            collection_obj.insert_many(record)
            return len(record)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        FILE_PATH = "Network_Data/phising.csv"  # Use forward slashes or raw strings for cross-platform paths
        DATABASE = "LOHITH"
        COLLECTION = "NetworkData"

        # Create object
        network_class = NetworkDataExtract()

        # Convert CSV to JSON records
        records = network_class.cv_to_json_convertor(FILE_PATH)
        print(f"Total records loaded: {len(records)}")

        # Insert into MongoDB
        inserted_count = network_class.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Total records inserted to MongoDB: {inserted_count}")

    except Exception as e:
        print("An error occurred during execution.")
        raise CustomException(e, sys)
