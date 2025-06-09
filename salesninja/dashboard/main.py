### Imports
import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja import data
from salesninja.ml import models, preprocessing, registry

from salesninja.params import *

### Main interface for SalesNinja dashboard
def preprocess(min_date:str = "2007-01-01", max_date:str = "2009-12-31"):
    """
    - Query Contoso dataset from our raw BigQuery dataset
    - If not existing locally, cache as csv files
    - Preprocess query data
    - Store processed data on our BigQuery cloud if it doesn't exist yet (truncate existing tables if older than $N_TIMESTEPS or number of tables higher than $N_TABLES)
    - Preprocessed data is not cached locally as csv (will be cached during training)
    """
    print(Fore.MAGENTA + "\n🌕 Preprocessing..." + Style.RESET_ALL)
    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    query = f"""
        SELECT {",".join(COLUMN_NAMES_DASHBOARD)}
        FROM `{GCP_SALESNINJA}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data = data.get_data_with_cache(
        query=query,
        gcp_project=GCP_SALESNINJA,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Process data
    X = data.drop("SalesAmount", axis=1)
    y = data[["SalesAmount"]]

    X_processed = preprocessing.preprocess_features(X)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    data_processed_with_timestamp = pd.DataFrame(np.concatenate((
        X_processed,
        y,
    ), axis=1))

    data.load_data_to_bq(
        data_processed_with_timestamp,
        gcp_project=GCP_SALESNINJA,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("[Main] preprocess() done \n")

def train():
    pass

def evaluate():
    pass

def predict():
    pass
