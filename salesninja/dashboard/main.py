### Imports
import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from salesninja.data import SalesNinja
from salesninja.ml import models, preprocessing, registry

from salesninja.params import *

from google.cloud import storage



### Main interface for SalesNinja dashboard
def preprocess(min_date:str = "2007-01-01", max_date:str = "2009-12-31"):
    """
    - Query Contoso dataset from our BigQuery dataset
    - If not existing locally, cache as csv files
    - Preprocess query data
    - Store processed data on our BigQuery cloud if it doesn't exist yet (truncate existing tables)
    - TO DO: delete tables if older than $N_TIMESTEPS or number of tables higher than $N_TABLES
    - Preprocessed data is not cached locally as csv (will be cached during training)
    """
    print(Fore.MAGENTA + "\n🌕 Preprocessing..." + Style.RESET_ALL)
    min_date = parse(min_date).strftime('%Y-%m-%d')
    max_date = parse(max_date).strftime('%Y-%m-%d')

    # SELECT {",".join(COLUMN_NAMES_DASHBOARD)}
    #FROM `{GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}`
    #FROM `nodal-clock-456815-g3.SalesNinja.data_ml_merged_10`
    query = f"""
        SELECT *
        FROM {GCP_SALESNINJA}.{BQ_DATASET}.data_ml_merged_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("queried",
            f"query_{min_date}_{max_date}_{int(DATA_SIZE*100)}.csv")
    data = SalesNinja().get_data_with_cache(
        query=query,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    data_clean = preprocessing.clean_data(data)

    # Process data
    X = data_clean.drop("SalesAmount", axis=1)
    y = data_clean[["SalesAmount"]]

    X_processed = preprocessing.preprocess_features(X, simple = False)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()

    data_processed = pd.DataFrame(np.concatenate((
    #data_processed = pd.DataFrame(pd.concat((
        X_processed,
        y,
    ), axis=1))

    data_processed.columns = X_processed.columns.append(pd.Index(["SalesAmount"]))

    SalesNinja().load_data_to_bq(
        data_processed,
        table=f'processed_{int(DATA_SIZE*100)}',
        truncate=True
    )

    print(Fore.MAGENTA + "\n🌖 preprocess() done \n" + Style.RESET_ALL)



def train(
        split_ratio: float = 0.10,
        forcenew = True
    ):
    """
    - Download processed data from BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """
    print(Fore.MAGENTA + "\n🌗 Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\n- Loading preprocessed validation data..." + Style.RESET_ALL)

    min_date = parse("2007-01-01").strftime('%Y-%m-%d') # e.g '2007-01-01'
    max_date = parse("2009-12-31").strftime('%Y-%m-%d') # e.g '2009-12-31'

    query = f"""
        SELECT *
        FROM `{GCP_SALESNINJA}`.{BQ_DATASET}.processed_{int(DATA_SIZE*100)}
        WHERE DateKey BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY DateKey
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{int(DATA_SIZE*100)}.csv")
    data_processed = SalesNinja().get_data_with_cache(
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("- Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    train_length = int(len(data_processed)*(1-split_ratio))

    data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1)#.to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1)#.to_numpy()

    X_train_processed = data_processed_train.iloc[:, :-1]
    y_train = data_processed_train.iloc[:, -1]

    X_val_processed = data_processed_val.iloc[:, :-1]
    y_val = data_processed_val.iloc[:, -1]

    X_train_processed = X_train_processed.drop("DateKey", axis = 1)
    X_val_processed = X_val_processed.drop("DateKey", axis = 1)

    # Train model using `models.py`
    model = registry.load_model()

    if (model is None) or (forcenew == True):
        model = models.initialize_model()

    model, history = models.train_model(
        model, X_train_processed, y_train, X_val_processed, y_val, test_size = split_ratio)

    val_mae = np.min(history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    registry.save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    registry.save_model(model=model)

    print(Fore.MAGENTA + "\n🌘 train() done \n" + Style.RESET_ALL)

    return val_mae



def evaluate(
        min_date:str = '2007-01-01',
        max_date:str = '2009-12-31',
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n🐱‍👤 Use case: evaluate" + Style.RESET_ALL)

    model = registry.load_model()
    assert model is not None

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT * EXCEPT(_0)
        FROM `{GCP_SALESNINJA}`.{BQ_DATASET}.processed_{int(DATA_SIZE*100)}
        WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
    """

    data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{min_date}_{max_date}_{int(DATA_SIZE*100)}.csv")
    data_processed = SalesNinja().get_data_with_cache(
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    #data_processed = data_processed.to_numpy()

    X_new = data_processed.iloc[:, :-1]
    X_new = X_new.drop("DateKey", axis = 1)
    y_new = data_processed.iloc[:, -1]

    #metrics_dict = models.evaluate_model(model=model, X=X_new, y=y_new)
    #mae = metrics_dict["mae"]
    mae = models.evaluate_model(model=model, X=X_new, y=y_new)

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    metrics_dict = {"mae": mae}
    registry.save_results(params=params, metrics=metrics_dict)

    print(Fore.MAGENTA + "\n🐱‍👤 evaluate() done \n" + Style.RESET_ALL)
    return mae



def predict(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the trained model
    """
    print(Fore.MAGENTA + "\n🐱‍👤 Use case: predict" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = SalesNinja().get_ml_data(min_date = "2009-01-01", max_date = "2009-12-31")
        X_pred = X_pred.drop("SalesAmount", axis = 1)

    model = registry.load_model()
    assert model is not None

    X_processed = preprocessing.preprocess_features(X_pred, simple = False)

    X_processed = X_processed.drop("DateKey", axis = 1)
    y_pred = model.predict(X_processed)

    print(Fore.MAGENTA + "\n🐱‍👤 prediction done: ", y_pred,
          " with a shape of ", y_pred.shape, "\n" + Style.RESET_ALL)
    return y_pred



if __name__ == '__main__':
    preprocess(min_date='2009-01-01', max_date='2015-01-01')
    train(split_ratio = 0.1)
    evaluate(min_date='2009-01-01', max_date='2015-01-01')
    predict()
