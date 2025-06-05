### Imports
##### loads a GPU-accelerated version of pandas, disable if this creates any problems!!
try:
    import cudf.pandas
    cudf.pandas.install()
    print("works!")
except:
    pass
#####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, learning_curve, LearningCurveDisplay

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from salesninja.ml.preprocessor import SimplePreprocessor





def initiate_model(X, y, modelclass = "xgb", preprocessor = "simple", test_size = 0.3):
    """
    Creates a new model, preprocesses data, creates a train/test split
    and fits the model it to the provided data. By default, uses XGBoost,
    simple preprocessing and a test size of 30%
    """
    if preprocessor == "simple":
        preprocess = SimplePreprocessor()
    else:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

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
