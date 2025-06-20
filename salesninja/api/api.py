import numpy as np
import pandas as pd

from pathlib import Path
from datetime import date

from salesninja.data import SalesNinja
import salesninja.ml.preprocessing as preprocessing
from salesninja.ml.registry import load_model#, load_synth
#from salesninja.ml.synthesis import SalesNinjaSynthesis
from salesninja.params import *

from typing import Annotated, Union
from fastapi import FastAPI, Query
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


# http://127.0.0.1:8899/predict?min_date=2010-01-01&max_date=2010-01-31
@app.get("/predict")
def predict(
        min_date: str, # 2009-01-01
        max_date: str, # 2010-12-31
        keylist: Annotated[Union[list[str], None], Query()] = None,
        valuelist: Annotated[Union[list[str], None], Query()] = None,
        grouplist: Annotated[Union[list[str], None], Query()] = None,
        debugfactor = 1 / (DATA_SIZE)
    ):
    """
    Returns a dictionary sorted by the days of the queried date range as keys
    and the forecast SalesAmount as values. If given a key- and valuelist each,
    will filter the actual or synthetic data by these keys before prediction
    (NOTE: currently not implemented). If given a grouplist, will group the
    actual or synthetic data by these columns (NOTE: currently not implemented).
    """

    daterange = pd.date_range(pd.to_datetime(min_date), pd.to_datetime(max_date), freq='d').to_list()

    if (date(min_date) < date("2010-01-01")) and (date(max_date) < date("2010-01-01")):
        query = f"""
        SELECT *
        FROM {GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
        """
        #{" ".join([f"AND {product} = {value}" for product, value in zip(keylist, valuelist)])}
        #{"" if grouplist is None else f"GROUP BY {', '.join([group for group in grouplist])}"}


        data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("queried",
            f"query_{min_date}_{max_date}_{int(DATA_SIZE*100)}.csv")

        newdata = SalesNinja().get_data_with_cache(
            query=query,
            cache_path=data_query_cache_path,
            data_has_header=True
        )
        newdata_processed = preprocessing.preprocess_features(newdata, simple = False)
        newdata_processed = preprocessing.seasonalize_data(newdata_processed)

    elif (date(min_date) < date("2010-01-01")):
        query = f"""
        SELECT *
        FROM {GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '2009-12-31'
        """

        data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("queried",
            f"query_{min_date}_'2009-12-31'_{int(DATA_SIZE*100)}.csv")

        newdata_actual = SalesNinja().get_data_with_cache(
            query=query,
            cache_path=data_query_cache_path,
            data_has_header=True
        )
        newdata_actual_processed = preprocessing.preprocess_features(newdata_actual, simple = False)
        newdata_actual_processed = preprocessing.seasonalize_data(newdata_actual_processed)
        newfacts = app.state.synth.create_facts("2010-01-01", max_date)
        newdata_synthetic = app.state.synth.sample_with_facts(newfacts)
        newdata_synthetic = newdata_synthetic.drop("DateKey", axis = 1)
        newdata = pd.concat([newdata_actual, newdata_synthetic], axis = 0)

    else:
        newfacts = app.state.synth.create_facts(min_date, max_date)
        newdata = app.state.synth.sample_with_facts(newfacts)

    newdata = newdata.drop("SalesAmount", axis = 1)
    newdata = newdata.drop("DateKey", axis = 1)
    prediction = (np.array(app.state.model.predict(newdata)) * debugfactor).tolist()

    results = dict(zip(daterange, prediction))

    return results



# http://127.0.0.1:8899/predict_basic?min_date=2009-01-01&max_date=2009-01-31
# http://127.0.0.1:8899/predict_basic?min_date=2009-01-01&max_date=2009-01-31&keylist=[ChannelKey,PromotionKey]&valuelist=1,1
# http://127.0.0.1:8899/predict_basic?min_date=2009-01-01&max_date=2009-01-31&grouplist=ProductKey
@app.get("/predict_basic")
def predict_basic(
        min_date: str, # 2007-01-01
        max_date: str, # 2009-12-31
        keylist: Annotated[Union[list[str], None], Query()] = None,
        valuelist: Annotated[Union[list[str], None], Query()] = None,
        grouplist: Annotated[Union[list[str], None], Query()] = None,
        debugfactor = int(1 / (DATA_SIZE))
    ):
    """
    Returns a dictionary sorted by the days of the queried date range as keys
    and the forecast SalesAmount as values. Only works on existing data (2007-2009).
    If given a key- and valuelist each, will filter the actual or synthetic data
    by these keys before prediction. If given a grouplist, will group the
    actual or synthetic data by these columns.
    """

    query = f"""
        SELECT *
        FROM {GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
    """
    #{"" if keylist is None else " ".join([f"AND {key} = 1" for key in keylist])}
    #{"" if grouplist is None else f"GROUP BY {', '.join([group for group in grouplist])}"}
    print(query)
    daterange = pd.date_range(pd.to_datetime(min_date, format = "%Y-%m-%d"),
                              pd.to_datetime(max_date, format = "%Y-%m-%d"),
                              freq='d').date.tolist()

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("queried",
            f"query_{min_date}_{max_date}_{keylist}_{valuelist}_{grouplist}_{int(DATA_SIZE*100)}.csv")

    newdata = SalesNinja().get_data_with_cache(
        query=query,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    newdata = newdata.drop("SalesAmount", axis = 1)

    newdata_processed = preprocessing.preprocess_features(newdata, simple = False)
    newdata_processed = preprocessing.seasonalize_data(newdata_processed)

    newdata_processed = newdata_processed.drop("DateKey", axis = 1)
    prediction = (np.array(app.state.model.predict(newdata_processed)) * debugfactor).tolist()

    results = dict(zip(daterange, prediction))

    return results
