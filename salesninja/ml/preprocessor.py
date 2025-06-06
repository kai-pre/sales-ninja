### Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer

from colorama import Fore, Style



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

    def fit_transform(df: pd.DataFrame) -> pd.DataFrame:
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
        df = df.rename(columns={'Weight': 'Weight(grams)'})

        # Delete the WeightUnitMeasureID column
        df = df.drop('WeightUnitMeasureID', axis=1)

        #4.
        df['CalendarQuarterLabel'] = df['CalendarQuarterLabel'].str.extract(r'(\d)').astype(int)

        #5.
        df = df.set_index('SalesKey')

        #6.
        df['DateKey'] = pd.to_datetime(df['DateKey'])

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
            'ProductSubcategoryKey', 'ClassID', 'StyleID', 'ColorID', 'Weight(grams)',
            'ProductCategoryKey', 'GeographyKey', 'EmployeeCount',
            'SellingAreaSize', 'sin_MonthNumber', 'cos_MonthNumber',
            'sin_CalendarDayOfWeekNumber', 'cos_CalendarDayOfWeekNumber',
            'sin_CalendarQuarterLabel', 'cos_CalendarQuarterLabel'
        ]

        rb_scaler = RobustScaler()
        df[cols_to_scale] = rb_scaler.fit_transform(df[cols_to_scale])

        ohe = OneHotEncoder(sparse_output=False)

        cols_to_encode = [
        'IsWorkDay','IsHoliday','BrandName','StoreType','GeographyType','RegionCountryName'
        ]
        ohe.fit(df[cols_to_encode])
        df[ohe.get_feature_names_out()] = ohe.transform(df[cols_to_encode])
        df = df.drop(columns=cols_to_encode)

        ### Boolean mask to remove outliers based on SalesAmount
        # Find the standard deviation of 'SalesAmount'
        std_sales = df['SalesAmount'].std()

        # Find 2.5 times the standard deviation
        threshold = 2.5 * std_sales

        # Create a boolean mask (True for values above threshold, False otherwise)
        boolean_mask = (df['SalesAmount']<=threshold)

        # Apply the boolean filtering
        df = df[boolean_mask]#.reset_index(drop=True) # Kai: not sure if index should be reset here...

        return df



def preprocess_features(X: pd.DataFrame, simple:bool = True) -> np.ndarray:
    print(Fore.BLUE + "\n🌖 Preprocessing features..." + Style.RESET_ALL)

    preprocessor = SimplePreprocessor() if simple else SalesNinjaPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("\n🌗 X processed, with shape ", X_processed.shape)

    return X_processed

def extract_prediction_facts(X: pd.DataFrame):
    """
    As a company has sparse information about the future, values are only predicted
    from these limited features. This function extracts these features and returns
    them.

    Facts are: DateKey, PromotionKey, DiscountPercent, CalendarYear, CalendarQuarterLabel,
    CalendarWeekLabel, IsWorkDay, IsHoliday, MonthNumber, CalendarDayOfWeekNumber
    """
    pass

def generate_prediction_estimates(X: pd.DataFrame, split = 0.3):
    """
    Splits a dataframe by the split ratio, and
    - extracts the prediction facts for the "test" set
    - generates estimates for selected features, based on the variable distribution
        of the "train" set.

    Estimates are based on the distribution in the "train" set and are bootstrapped
    (and stratified?).

    Facts are: DateKey, PromotionKey, DiscountPercent, CalendarYear, CalendarQuarterLabel,
    CalendarWeekLabel, IsWorkDay, IsHoliday, MonthNumber, CalendarDayOfWeekNumber

    Estimates are: 'channelKey', 'StoreKey', 'ProductKey',
       'UnitCost', 'UnitPrice', 'SalesQuantity', 'ReturnQuantity',
       'ReturnAmount', 'DiscountQuantity', 'DiscountAmount', 'TotalCost',
       'SalesAmount', 'CalendarYear', 'CalendarQuarterLabel', 'CalendarWeekLabel',
       'IsWorkDay', 'IsHoliday', 'MonthNumber', 'CalendarDayOfWeekNumber',
       'ProductSubcategoryKey', 'BrandName', 'ClassID', 'StyleID', 'ColorID',
       'Weight', 'WeightUnitMeasureID', 'StockTypeID', 'ProductCategoryKey',
       'GeographyKey', 'StoreType', 'EmployeeCount', 'SellingAreaSize',
       'GeographyType', 'ContinentName', 'CityName', 'StateProvinceName',
       'RegionCountryName'
    """
    pass
