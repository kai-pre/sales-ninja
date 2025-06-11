### Imports
##### loads a GPU-accelerated version of pandas, disable if this creates any problems!!
"""
print("Trying CUDA/CUDF import ...")
try:
    import cudf.pandas
    cudf.pandas.install()
    print("CUDA/CUDF found, using CUDF.pandas")
except:
    # disable tensorflow warnings
    TF_CPP_MIN_LOG_LEVEL=3
    print("CUDA/CUDF not found, using standard pandas and disabling tensorflow warnings")
"""
#####
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import sys

from colorama import Fore, Style

from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, learning_curve, LearningCurveDisplay

#from sklearn.pipeline import Pipeline

#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
import salesninja.ml.preprocessing as preprocessing

from salesninja.params import *





def quickstart_model(X, y, modelclass = "xgb", preprocessor = "simple", test_size = 0.3):
    """
    Creates a new model, preprocesses data, creates a train/test split
    and fits the model it to the provided data. By default, uses XGBoost,
    simple preprocessing and a test size of 30%
    """
    if preprocessor == "simple":
        preprocess = preprocessing.SimplePreprocessor()
    else:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, shuffle = False)

    X_train_proc = preprocess.fit_transform(X_train)
    X_test_proc = preprocess.transform(X_test)

    if modelclass == "xgb":
        model = XGBRegressor(max_depth=10, n_estimators=200, learning_rate=0.1, device = "cuda")
        model.fit(X_train, y_train,
        # evaluate loss at each iteration
        eval_set=[(X_train, y_train), (X_test, y_test)]#,
        # stop iterating when eval loss increases 5 times in a row
        # early_stopping_rounds=5
        )
    else:
        return None
    return model


def initialize_model(modelclass = "xgb", n_estimators = 500, device = "cuda"):
    """
    Initializes a new model, XGBoost by default
    """
    if modelclass == "xgb":
        model = XGBRegressor(max_depth=10, n_estimators=n_estimators, learning_rate=0.1,
                             # stop iterating when eval loss increases 5 times in a row
                             enable_categorical=True,
                             early_stopping_rounds=5,
                             eval_metric=mean_absolute_error,
                             n_jobs=-1, device = device,
                             )
    else:
        return None
    return model


def train_model(model, X, y, X_val = None, y_val = None, test_size = 0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\n[ML] Training model..." + Style.RESET_ALL)

    # preprocessing is already done in main
    # X_prep = preprocessing.preprocess_features(X)

    if X_val is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, shuffle = False)
        model.fit(X_train, y_train,
            # evaluate loss at each iteration
            eval_set=[(X_train, y_train), (X_test, y_test)],
            xgb_model = model,
            verbose = False
        )
    else:
        model.fit(X, y,
            # evaluate loss at each iteration
            eval_set=[(X, y), (X_val, y_val)],
            verbose = False
        )

    results = model.evals_result()
    history = dict(rmse = results["validation_0"]["rmse"],
                   val_rmse = results["validation_1"]["rmse"],
                   mae = results["validation_0"]["mean_absolute_error"],
                   val_mae = results["validation_1"]["mean_absolute_error"])

    trainingrows = len(X_train) if X_val is None else len(X)
    print(f"✅ Model trained on {trainingrows} rows with min val MAE: {round(np.min(history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
    ):
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\n[ML] Evaluating model on {NUMBER_OF_ROWS} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n[ML] No model to evaluate")
        return None

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"\n[ML] Model evaluated, MAE: {round(mae, 2)}")

    return mae


def load_XGB_model(filepath):
    model = initialize_model()
    model.load_model(filepath)
    return model
