import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja import data
import salesninja.ml.preprocessing as preprocessing
from salesninja.ml.registry import load_model
import salesninja.ml.synthesis as synthesis

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.state.model = load_model()
assert app.state.model is not None


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



# http://127.0.0.1:8000/
@app.get("/")
def root():
    return {'testing': "The API works!"}


# http://127.0.0.1:8000/predict?starttime=2007-01-01&endtime=2009-12-31
@app.get("/get_db_data")
def get_db_data(
        starttime: str = "2007-01-01",
        endtime: str = "2009-12-31",
    ):
    """
    Returns merged, pre-filtered data for the dashboard
    Assumes "starttime" and "endtime" is provided as a string by the user in "%Y-%m-%d" format
    """
    querieddata = data.SalesNinja().get_db_data(starttime, endtime)

    return querieddata.to_dict()


"""
def predict(
        starttime: str, # 2007-01-01
        endtime: str, # 2009-12-31
    ):

    # get synthesis model
    # synthesize data for queried timerange
    # use prediction model to predict a y for the synthesized data


    prediction = app.state.model.predict(synthdata)

    return prediction
"""
