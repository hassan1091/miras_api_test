from typing import Union

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel
# 
from utils.csv_to_json import read_old_factory_data_to_json


app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/old_data")
def read_old_data():
    return JSONResponse(content=read_old_factory_data_to_json())
