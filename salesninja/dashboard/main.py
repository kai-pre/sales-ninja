### Imports
import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja import data
from salesninja.ml import *

### Main interface for SalesNinja dashboard
def load_and_preprocess(min_date:str = "2007-01-01", max_data:str = "2009-12-31"):
    """
    - Query Contoso dataset from our raw BigQuery dataset
    - If not existing locally, cache as csv files
    - Preprocess query data
    - Store processed data on our BigQuery cloud if it doesn't exist yet (truncate existing tables if older than $N_TIMESTEPS or higher than $N_TABLES)
    - Preprocessed data is not cached locally as csv (will be cached during training)
    """
    print(Fore.MAGENTA + "\n 🌑 Preprocessing..." + Style.RESET_ALL)
    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    ##### TO DO
    query = f"""
        SELECT {",".join(COLUMN_NAMES_DASHBOARD)}
        FROM `{GCP_SALESNINJA}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
    """

def train():
    pass

def evaluate():
    pass

def predict():
    pass
