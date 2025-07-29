from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

uri = "mongodb+srv://lohithhs1414:1234Admin@ac-hz8azsi.ezjvzxl.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
    print("Databases:", client.list_database_names())

except ConnectionFailure as e:
    print("❌ Could not connect to MongoDB Atlas:", e)
