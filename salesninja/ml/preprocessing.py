### Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer

from colorama import Fore, Style


def clean_data(df):
    """
    Removes outliers based on "SalesAmount"
    """
    ### Boolean mask to remove outliers based on SalesAmount
    # Find the standard deviation of 'SalesAmount'
    std_sales = df['SalesAmount'].std()

    # Find 2.5 times the standard deviation
    threshold = 2.5 * std_sales

    # Create a boolean mask (True for values above threshold, False otherwise)
    boolean_mask = (df['SalesAmount'] <= threshold)

    # Apply the boolean filtering
    df = df[boolean_mask]#.reset_index(drop=True) # Kai: not sure if index should be reset here...

    return df



class SimplePreprocessor():
    """
    Simple preprocessor to get started with ML quickly
    """
    def __init__(self):
        self.preprocessor = make_column_transformer([RobustScaler(), make_column_selector(dtype_include = "number")], remainder="passthrough")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = self.preprocessor.fit_transform(df)
        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = self.preprocessor.transform(df)
        return df_processed



class SalesNinjaPreprocessor():
    """
    Custom preprocessor for Contoso dataset
    """
    def __init__(self):
        pass

    def fit_transform(self, df: pd.DataFrame, debug = False) -> pd.DataFrame:
        if "Unnamed: 0" in df.columns:
            if debug:
                print("Column 'Unnamed: 0' found, dropped!")
            df = df.drop("Unnamed: 0", axis = 1)

        # Instantiate a SimpleImputer object with your strategy of choice
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        # Call the "fit" method on the object
        imputer.fit(df[['ProductSubcategoryKey']])
        df['ProductSubcategoryKey'] = imputer.transform(df[['ProductSubcategoryKey']])
        imputer.fit(df[['ClassID']])
        df['ClassID'] = imputer.transform(df[['ClassID']])
        imputer.fit(df[['StyleID']])
        df['StyleID'] = imputer.transform(df[['StyleID']])  # Fixed this line
        imputer.fit(df[['ColorID']])
        df['ColorID'] = imputer.transform(df[['ColorID']])
        imputer.fit(df[['Weight']])
        df['Weight'] = imputer.transform(df[['Weight']])
        imputer.fit(df[['StockTypeID']])
        df['StockTypeID'] = imputer.transform(df[['StockTypeID']])
        imputer.fit(df[['ProductCategoryKey']])
        df['ProductCategoryKey'] = imputer.transform(df[['ProductCategoryKey']])

        # delete blank spaces in columns
        df['DateKey'] = pd.to_datetime(df['DateKey'], format = '%Y-%m-%d').dt.date

        df = df.apply(lambda x: x.str.strip() if (x.dtype == "object") and (x.name != "DateKey") else x)

        df['BrandName'] = df['BrandName'].fillna('N/A')


        #Basic column transforms
        #1. Promo key 1 means no discount. ALl other promo keys indicate some discount. We change promo keys to 1 and 0. 1 means no disc, 0 means disc
        #2. Stock type ID 1 means High, 3 means low. THis is opposite of how the ML model learns. So, we switch
        #3. Weights are in different units. Convert all to grams.
        #4. Convert quarter values from Q1, Q2...to 1,2...This is neeeded for cyclical engg. later
        #5. Set Sales Key as index of dataframe
        #6. Ensure DateKey column is in DateTime format

        #1.
        df['PromotionKey'] = np.where(df['PromotionKey'] == 1, 1, 0)

        #2.
        df['StockTypeID'] = df['StockTypeID'].replace({1: 3, 3: 1})

        #3.
        conversion_factors = {
            'pounds': 453.592,
            'ounces': 28.3495,
            'grams': 1.0
        }

        # Convert weights to grams
        df['Weight'] = df['Weight'] * df['WeightUnitMeasureID'].map(conversion_factors)

        # Rename the Weight column
        df = df.rename(columns={'Weight': 'Weight_grams'})

        # Delete the WeightUnitMeasureID column
        df = df.drop('WeightUnitMeasureID', axis=1)

        #4.
        df['CalendarQuarterLabel'] = df['CalendarQuarterLabel'].str.extract(r'(\d)').astype(int)

        #5.
        #df = df.set_index('SalesKey')

        #Cyclical engineering to ensure proximity of dec to jan
        months_in_year = 12
        df['sin_MonthNumber'] = np.sin(2*np.pi*df.MonthNumber/months_in_year)
        df['cos_MonthNumber'] = np.cos(2*np.pi*df.MonthNumber/months_in_year)
        df.drop(columns=['MonthNumber'], inplace=True)
        df.columns

        #Cyclical engineering to ensure proximity of sat to sun usw.
        #calendarweeklabel dropped as its impact is same as cyclical engg.
        days_in_week = 7
        df['sin_CalendarDayOfWeekNumber'] = np.sin(2*np.pi*df.CalendarDayOfWeekNumber/days_in_week)
        df['cos_CalendarDayOfWeekNumber'] = np.cos(2*np.pi*df.CalendarDayOfWeekNumber/days_in_week)
        df.drop(columns=['CalendarDayOfWeekNumber','CalendarWeekLabel'], inplace=True)
        df.columns

        #Cyclical engineering to ensure proximity of Q1 to Q4 usw.
        quarters_in_year = 4
        df['sin_CalendarQuarterLabel'] = np.sin(2*np.pi*df.CalendarQuarterLabel/quarters_in_year)
        df['cos_CalendarQuarterLabel'] = np.cos(2*np.pi*df.CalendarQuarterLabel/quarters_in_year)
        df.drop(columns=['CalendarQuarterLabel'], inplace=True)
        df.columns

        rb_scaler = RobustScaler()

        cols_to_scale = [
            'channelKey', 'StoreKey', 'ProductKey', 'PromotionKey','StockTypeID',
            'UnitCost', 'UnitPrice', 'TotalCost', 'ReturnQuantity', 'ReturnAmount',
            'DiscountQuantity', 'DiscountAmount', 'DiscountPercent', 'CalendarYear',
            'ProductSubcategoryKey', 'ClassID', 'StyleID', 'ColorID', 'Weight_grams',
            'ProductCategoryKey', 'GeographyKey', 'EmployeeCount',
            'SellingAreaSize', 'sin_MonthNumber', 'cos_MonthNumber',
            'sin_CalendarDayOfWeekNumber', 'cos_CalendarDayOfWeekNumber',
            'sin_CalendarQuarterLabel', 'cos_CalendarQuarterLabel'
        ]

        rb_scaler = RobustScaler()
        df[cols_to_scale] = rb_scaler.fit_transform(df[cols_to_scale])

        ohe = OneHotEncoder(sparse_output=False)

        cols_to_encode = [
            'IsWorkDay','BrandName','StoreType','GeographyType','RegionCountryName'
        ]
        ohe.fit(df[cols_to_encode])
        df[ohe.get_feature_names_out()] = ohe.transform(df[cols_to_encode])
        df = df.drop(columns=cols_to_encode)

        labelencoder = LabelEncoder()
        df["CityName"] = labelencoder.fit_transform(df["CityName"])
        df["ContinentName"] = labelencoder.fit_transform(df["ContinentName"])
        df["StateProvinceName"] = labelencoder.fit_transform(df["StateProvinceName"])

        # additional cleaning for BigQuery compatibility
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('.', '')
        df.columns = df.columns.str.replace('/', '')

        if "int64_field_0" in df.columns:
            if debug:
                print("Column 'int64_field_0' found, dropped!")
            df = df.drop('int64_field_0', axis = 1)

        return df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)



def preprocess_features(X: pd.DataFrame, simple:bool = True) -> np.ndarray:
    print(Fore.BLUE + "\n[preprocessing] Preprocessing features..." + Style.RESET_ALL)

    preprocessor = SimplePreprocessor() if simple else SalesNinjaPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    if simple:
        X_processed = pd.DataFrame(X_processed)
        X_processed.columns = X.columns
    print("\n[preprocessing] X processed, with shape ", X_processed.shape)

    return X_processed



def seasonalize(data: pd.DataFrame, season = "daily") -> pd.DataFrame:
    """
    Aggregates data by season. "season" can be "daily", "weekly", "monthly",
    "quarterly", or "yearly"
    """
    aggfeatures = {"UnitCost": "mean", "UnitPrice": "mean", "ReturnQuantity": "sum", "ReturnAmount": "sum",
               "DiscountQuantity": "sum", "DiscountAmount": "sum", "DiscountPercent": "mean", "IsWorkDay": "mean",
               "IsHoliday": "mean", "Weight": "sum", "EmployeeCount": "mean", "SellingAreaSize": "mean",
               "TotalCost": "sum", "SalesQuantity": "sum", "SalesAmount": "sum"}

    match season:
        case "daily":
            data_season = data.groupby("DateKey").agg(aggfeatures)
        case "weekly":
            data_season = data.groupby(["CalendarYear", "CalendarWeekLabel"]).agg(aggfeatures).reset_index()
        case "monthly":
            data_season = data.groupby(["CalendarYear", "MonthNumber"]).agg(aggfeatures).reset_index()
        case "quarterly":
            data_season = data.groupby(["CalendarYear", "CalendarQuarterLabel"]).agg(aggfeatures).reset_index()
        case "yearly":
            data_season = data_yearly = data.groupby("CalendarYear").agg(aggfeatures)

    return data_season



def extract_prediction_facts(X: pd.DataFrame, with_datekey = False):
    """
    As a company has sparse information about the future, values are only predicted
    from these limited features. This function extracts these features and returns
    them.

    Facts are: DateKey, CalendarYear, CalendarQuarterLabel, CalendarWeekLabel,
    IsWorkDay, MonthNumber, CalendarDayOfWeekNumber

    PromotionKey and DiscountPercent could be considered as well, but would necessitate
    much more user input.

    CalendarYear is considered redundant, and would probably mess with data synthesis
    if included.
    """
    if with_datekey:
        return X[["DateKey", "CalendarQuarterLabel",
        "CalendarWeekLabel", "IsWorkDay", "MonthNumber", "CalendarDayOfWeekNumber"]]
    return X[["CalendarQuarterLabel",
        "CalendarWeekLabel", "IsWorkDay", "MonthNumber", "CalendarDayOfWeekNumber"]]


if __name__ == "__main__":
    pass


"""import pandas as pd
a = pd.DataFrame({"a": [1, 8, 1], "b": [2, 5, 2], "c": [3, -5, 3], "d": [4, 9, 4]})
b = pd.DataFrame({"a": [1], "b": [2]})

a.merge(b).sample(1)
"""
