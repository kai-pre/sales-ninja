### Imports
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector



class SimplePreprocessor():
    """
    Simple preprocessor to get started with ML quickly
    """
    def __init__():
        make_column_transformer([RobustScaler(), make_column_selector(dtype_include = "number")])

class SalesNinjaPreprocessor():
    """
    Custom preprocessor for Contoso dataset
    """
    def __init__():
        pass
