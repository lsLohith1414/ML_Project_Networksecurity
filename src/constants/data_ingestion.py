
# conecting the DB
# data base name
# DB collection name --> tables names



# storing the data after data ingestion, 
# means storing the data after ingestion like train data test data, in some folders, this 
# folders we took as a constants

# data_ingestion -->main ingestion files are saved here
# features_store --> store the raw data like phishingData.csv
# ingested --> store the ingested data like train.csv, test.csv

# Defining the folder constants

DATA_INGESTION_DATABASE_NAME: str = "LOHITH"
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_NAME: str = "features_store"
DATA_INGESTION_INGESTED_NAME: str = "ingested" 
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: str = 0.2