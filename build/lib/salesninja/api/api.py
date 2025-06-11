import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja.data import SalesNinja
import salesninja.ml.preprocessing as preprocessing
from salesninja.ml.registry import load_model#, load_synth
from salesninja.ml.synthesis import SalesNinjaSynthesis
from salesninja.params import *

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.state.model = load_model()
assert app.state.model is not None

#app.state.synth = load_synth()
#assert app.state.synth is not None


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
    querieddata = SalesNinja().get_db_data(min_date, max_date)

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

    querieddata = SalesNinja().get_ml_data(min_date, max_date)

    return querieddata.to_dict()


@app.get("/predict")
def predict(
        min_date: str, # 2007-01-01
        max_date: str, # 2009-12-31
    ):

    daterange = pd.date_range(pd.to_datetime(min_date), pd.to_datetime(max_date), freq='d').to_list()

    newfacts = app.state.synth.create_facts(min_date, max_date)
    newdata = app.state.synth.sample_with_facts(newfacts)

    # TO DO: preprocess this

    # use prediction model to predict a y for the synthesized data
    prediction = app.state.model.predict(newdata)

    return dict(DateKey = daterange, SalesAmount = prediction)


# http://127.0.0.1:8899/predict_basic?min_date=2009-01-01&max_date=2009-01-31
@app.get("/predict_basic")
def predict_basic(
        min_date: str, # 2007-01-01
        max_date: str, # 2009-12-31
    ):

    query = f"""
        SELECT *
        FROM {GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
    """
    daterange = pd.date_range(pd.to_datetime(min_date, format = "%Y-%m-%d"),
                              pd.to_datetime(max_date, format = "%Y-%m-%d"),
                              freq='d').date.tolist()

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("queried",
            f"query_{min_date}_{max_date}_{int(DATA_SIZE*100)}.csv")

    newdata = SalesNinja().get_data_with_cache(
        query=query,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    newdata = newdata.drop("SalesAmount", axis = 1)

    newdata_processed = preprocessing.preprocess_features(newdata, simple = False)

    newdata_processed = newdata_processed.drop("DateKey", axis = 1)
    prediction = app.state.model.predict(newdata_processed).tolist()

    results = dict(zip(daterange, prediction))
    print(results)

    return results
