import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja import data
import salesninja.ml.preprocessing as preprocessing
from salesninja.ml.registry import load_model, load_synth
from salesninja.ml.synthesis import SalesNinjaSynthesis

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.state.model = load_model()
assert app.state.model is not None

app.state.synth = load_synth()
assert app.state.synth is not None


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


# http://127.0.0.1:8000/get_db_data?min_date=2007-01-01&max_date=2009-12-31
@app.get("/get_db_data")
def get_db_data(
        min_date: str = "2007-01-01",
        max_date: str = "2009-12-31",
    ):
    """
    Returns merged, pre-filtered data for the dashboard
    Assumes "starttime" and "endtime" is provided as a string by the user in "%Y-%m-%d" format
    """
    querieddata = data.SalesNinja().get_db_data(min_date, max_date)

    return querieddata.to_dict()


# http://127.0.0.1:8000/get_ml_data?min_date=2007-01-01&max_date=2009-12-31
@app.get("/get_ml_data")
def get_ml_data(
        min_date: str = "2007-01-01",
        max_date: str = "2009-12-31",
    ):
    """
    Returns merged, pre-filtered data for machine learning
    Assumes "starttime" and "endtime" is provided as a string by the user in "%Y-%m-%d" format
    """

    querieddata = data.SalesNinja().get_ml_data(min_date, max_date)

    return querieddata.to_dict()


@app.get("/predict")
def predict(
        min_date: str, # 2007-01-01
        max_date: str, # 2009-12-31
    ):

    newfacts = app.state.synth.create_facts(min_date, max_date)
    newdata = app.state.synth.sample_with_facts(newfacts)

    # TO DO: preprocess this

    # use prediction model to predict a y for the synthesized data
    prediction = app.state.model.predict(newdata)

    return prediction.to_dict()
