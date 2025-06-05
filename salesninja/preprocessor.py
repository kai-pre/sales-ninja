### Imports
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector



class SimplePreprocessor():
    def __init__():
        make_column_transformer([RobustScaler(), make_column_selector(dtype_include = "number")])

class SalesNinjaPreprocessor():
    def __init__():
        pass
