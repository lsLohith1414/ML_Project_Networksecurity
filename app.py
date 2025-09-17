from src.pipelines.training_pipeline import TrainingPipeline
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging



import os
import sys

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
import pymongo



from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File, UploadFile,Request
from uvicorn import run as app_run 
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from src.utilites.main_utils.utils import load_object
from src.constants.data_ingestion import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME  
from src.utilites.ml_utils.model.estimetor import NetworkModel

client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile = ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collections = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags= ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        model_train = TrainingPipeline()
        artifacts = model_train.run_training_pipline()
        return Response("Training is successfull")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_models/preprocessor.pkl") 
        final_model=load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise NetworkSecurityException(e,sys)

    


if __name__ == "__main__":

    app_run(app=app, host="localhost",port=8000)