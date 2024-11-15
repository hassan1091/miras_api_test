import json
import os
import sys
from typing import Union

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
# 
from utils.csv_to_json import read_old_factory_data_to_json
from predict import predict


#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
    
class DeviceData(BaseModel):
    temperature_c: float = Field(examples=[30])
    humidity_percent: float= Field(examples=[37.13])
    pressure_pa: int= Field(examples=[101494])
    operating_hours: int= Field(examples=[2335])
    response_time_ms: int= Field(examples=[600])
    

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/old_data")
def read_old_data():
    return JSONResponse(content=read_old_factory_data_to_json())

@app.get("/predict")
def read_predict():
    input_data = {
        "temperature_c": 30,
        "humidity_percent": 37.13,
        "pressure_pa": 101494,
        "operating_hours": 2335,
        "response_time_ms": 600
    }
    return JSONResponse(content={"input": input_data,"predictions": predict(input_data)}
)

@app.post("/predict")
def predict(data:DeviceData):
    input_data = data.model_dump()
    return JSONResponse(content={"input": input_data,"predictions": predict(input_data)}
)


def listen():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    listen()