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



def seasonalize_data(data: pd.DataFrame, season = "daily") -> pd.DataFrame:
    """
    Aggregates data by season. "season" can be "daily", "weekly", "monthly",
    "quarterly", or "yearly"
    """
    print(Fore.BLUE + f"\n[preprocessing] Seasonalizing features, {season} ..." + Style.RESET_ALL)
    # raw:
    # SalesKey,DateKey,channelKey,StoreKey,ProductKey,PromotionKey,UnitCost,UnitPrice,
    # SalesQuantity,ReturnQuantity,ReturnAmount,DiscountQuantity,DiscountAmount,TotalCost,
    # SalesAmount,DiscountPercent,CalendarYear,CalendarQuarterLabel,CalendarWeekLabel,IsWorkDay,
    # MonthNumber,CalendarDayOfWeekNumber,ProductSubcategoryKey,BrandName,ClassID,StyleID,ColorID,
    # Weight,WeightUnitMeasureID,StockTypeID,ProductCategoryKey,GeographyKey,StoreType,EmployeeCount,
    # SellingAreaSize,GeographyType,ContinentName,CityName,StateProvinceName,RegionCountryName

    # after preprocessing:
    # SalesKey,DateKey,channelKey,StoreKey,ProductKey,PromotionKey,UnitCost,UnitPrice,
    # SalesQuantity,ReturnQuantity,ReturnAmount,DiscountQuantity,DiscountAmount,TotalCost,
    # DiscountPercent,CalendarYear,ProductSubcategoryKey,ClassID,StyleID,ColorID,
    # Weight_grams,StockTypeID,ProductCategoryKey,GeographyKey,EmployeeCount,SellingAreaSize,
    # ContinentName,CityName,StateProvinceName,sin_MonthNumber,cos_MonthNumber,
    # sin_CalendarDayOfWeekNumber,cos_CalendarDayOfWeekNumber,sin_CalendarQuarterLabel,
    # cos_CalendarQuarterLabel,IsWorkDay_WeekEnd,IsWorkDay_WorkDay,BrandName_A_Datum,
    # BrandName_Adventure_Works,BrandName_Contoso,BrandName_Fabrikam,BrandName_Litware,
    # BrandName_NA,BrandName_Northwind_Traders,BrandName_Proseware,BrandName_Southridge_Video,
    # BrandName_Tailspin_Toys,BrandName_The_Phone_Company,BrandName_Wide_World_Importers,
    # StoreType_Catalog,StoreType_Online,StoreType_Reseller,StoreType_Store,GeographyType_City,
    # RegionCountryName_Armenia,RegionCountryName_Australia,RegionCountryName_Bhutan,
    # RegionCountryName_Canada,RegionCountryName_China,RegionCountryName_Denmark,
    # RegionCountryName_France,RegionCountryName_Germany,RegionCountryName_Greece,
    # RegionCountryName_India,RegionCountryName_Iran,RegionCountryName_Ireland,
    # RegionCountryName_Italy,RegionCountryName_Japan,RegionCountryName_Kyrgyzstan,
    # RegionCountryName_Malta,RegionCountryName_Pakistan,RegionCountryName_Poland,
    # RegionCountryName_Portugal,RegionCountryName_Romania,RegionCountryName_Russia,
    # RegionCountryName_Singapore,RegionCountryName_Slovenia,RegionCountryName_South_Korea,
    # RegionCountryName_Spain,RegionCountryName_Sweden,RegionCountryName_Switzerland,
    # RegionCountryName_Syria,RegionCountryName_Taiwan,RegionCountryName_Thailand,
    # RegionCountryName_Turkmenistan,RegionCountryName_United_Kingdom,RegionCountryName_United_States,
    # RegionCountryName_the_Netherlands,SalesAmount

    # kept after seasonalizing:
    # DateKey,UnitCost,UnitPrice,SalesQuantity,ReturnQuantity,ReturnAmount,
    # DiscountQuantity,DiscountAmount,TotalCost,DiscountPercent,CalendarYear,
    # Weight_grams,(sin_MonthNumber,cos_MonthNumber),(sin_CalendarDayOfWeekNumber,
    # cos_CalendarDayOfWeekNumber),(sin_CalendarQuarterLabel,cos_CalendarQuarterLabel),
    # IsWorkDay_WeekEnd,IsWorkDay_WorkDay,BrandName_A_Datum,BrandName_Adventure_Works,
    # BrandName_Contoso,BrandName_Fabrikam,BrandName_Litware,BrandName_NA,
    # BrandName_Northwind_Traders,BrandName_Proseware,BrandName_Southridge_Video,
    # BrandName_Tailspin_Toys,BrandName_The_Phone_Company,BrandName_Wide_World_Importers,
    # StoreType_Catalog,StoreType_Online,StoreType_Reseller,StoreType_Store,
    # GeographyType_City,RegionCountryName_Armenia,RegionCountryName_Australia,
    # RegionCountryName_Bhutan,RegionCountryName_Canada,RegionCountryName_China,
    # RegionCountryName_Denmark,RegionCountryName_France,RegionCountryName_Germany,
    # RegionCountryName_Greece,RegionCountryName_India,RegionCountryName_Iran,
    # RegionCountryName_Ireland,RegionCountryName_Italy,RegionCountryName_Japan,
    # RegionCountryName_Kyrgyzstan,RegionCountryName_Malta,RegionCountryName_Pakistan,
    # RegionCountryName_Poland,RegionCountryName_Portugal,RegionCountryName_Romania,
    # RegionCountryName_Russia,RegionCountryName_Singapore,RegionCountryName_Slovenia,
    # RegionCountryName_South_Korea,RegionCountryName_Spain,RegionCountryName_Sweden,
    # RegionCountryName_Switzerland,RegionCountryName_Syria,RegionCountryName_Taiwan,
    # RegionCountryName_Thailand,RegionCountryName_Turkmenistan,RegionCountryName_United_Kingdom,
    # RegionCountryName_United_States,RegionCountryName_the_Netherlands,SalesAmount

    aggfeatures = {
        "UnitCost": "sum", "UnitPrice": "sum", "ReturnQuantity": "sum",
        "ReturnAmount": "sum", "DiscountQuantity": "sum", "DiscountAmount": "sum",
        "DiscountPercent": "mean", "CalendarYear": "first", "Weight_grams": "sum",
        "IsWorkDay_WeekEnd": "mean","IsWorkDay_WorkDay": "mean",
        "BrandName_A_Datum": "mean", "BrandName_Adventure_Works": "mean",
        "BrandName_Contoso": "mean", "BrandName_Fabrikam": "mean",
        "BrandName_Litware": "mean", "BrandName_NA": "mean",
        "BrandName_Northwind_Traders": "mean", "BrandName_Proseware": "mean",
        "BrandName_Southridge_Video": "mean", "BrandName_Tailspin_Toys": "mean",
        "BrandName_The_Phone_Company": "mean", "BrandName_Wide_World_Importers": "mean",
        "StoreType_Catalog": "mean", "StoreType_Online": "mean", "StoreType_Reseller": "mean",
        "StoreType_Store": "mean", "GeographyType_City": "mean",
        "RegionCountryName_Armenia": "mean", "RegionCountryName_Australia": "mean",
        "RegionCountryName_Bhutan": "mean", "RegionCountryName_Canada": "mean",
        "RegionCountryName_China": "mean",
        "RegionCountryName_Denmark": "mean", "RegionCountryName_France": "mean",
        "RegionCountryName_Germany": "mean", "RegionCountryName_Greece": "mean",
        "RegionCountryName_India": "mean", "RegionCountryName_Iran": "mean",
        "RegionCountryName_Ireland": "mean", "RegionCountryName_Italy": "mean",
        "RegionCountryName_Japan": "mean", "RegionCountryName_Kyrgyzstan": "mean",
        "RegionCountryName_Malta": "mean", "RegionCountryName_Pakistan": "mean",
        "RegionCountryName_Poland": "mean", "RegionCountryName_Portugal": "mean",
        "RegionCountryName_Romania": "mean", "RegionCountryName_Russia": "mean",
        "RegionCountryName_Singapore": "mean", "RegionCountryName_Slovenia": "mean",
        "RegionCountryName_South_Korea": "mean", "RegionCountryName_Spain": "mean",
        "RegionCountryName_Sweden": "mean", "RegionCountryName_Switzerland": "mean",
        "RegionCountryName_Syria": "mean", "RegionCountryName_Taiwan": "mean",
        "RegionCountryName_Thailand": "mean", "RegionCountryName_Turkmenistan": "mean",
        "RegionCountryName_United_Kingdom": "mean", "RegionCountryName_United_States": "mean",
        "RegionCountryName_the_Netherlands": "mean", "SalesQuantity": "sum"}

    if "SalesAmount" in data.columns:
        aggfeatures.update({"SalesAmount": "sum"})

    match season:
        case "daily":
            aggfeatures.update({"sin_MonthNumber": "first","cos_MonthNumber": "first",
                             "sin_CalendarDayOfWeekNumber": "first",
                             "cos_CalendarDayOfWeekNumber": "first",
                             "sin_CalendarQuarterLabel": "first",
                             "cos_CalendarQuarterLabel": "first"})
            data_season = data.groupby("DateKey").agg(aggfeatures)
        case "weekly":
            aggfeatures.update({"sin_MonthNumber": "first","cos_MonthNumber": "first",
                             "sin_CalendarQuarterLabel": "first",
                             "cos_CalendarQuarterLabel": "first"})
            data_season = data.groupby(["CalendarYear", "CalendarWeekLabel"]).agg(aggfeatures).reset_index()
        case "monthly":
            aggfeatures.update({"sin_MonthNumber": "first","cos_MonthNumber": "first",
                             "sin_CalendarQuarterLabel": "first",
                             "cos_CalendarQuarterLabel": "first"})
            data_season = data.groupby(["CalendarYear", "MonthNumber"]).agg(aggfeatures).reset_index()
        case "quarterly":
            aggfeatures.update({"sin_CalendarQuarterLabel": "first",
                             "cos_CalendarQuarterLabel": "first"})
            data_season = data.groupby(["CalendarYear", "CalendarQuarterLabel"]).agg(aggfeatures).reset_index()
        case "yearly":
            data_season = data.groupby("CalendarYear").agg(aggfeatures)

    data_season.reset_index(inplace = True)
    return data_season



def seasonalize_y(data: pd.DataFrame, season = "daily") -> pd.Series:
    match season:
        case "daily":
            data_season = data.groupby("DateKey").agg({"SalesAmount": "sum"})
        case "weekly":
            data_season = data.groupby(["CalendarYear", "CalendarWeekLabel"]).agg({"SalesAmount": "sum"})
        case "monthly":
            data_season = data.groupby(["CalendarYear", "MonthNumber"]).agg({"SalesAmount": "sum"})
        case "quarterly":
            data_season = data.groupby(["CalendarYear", "CalendarQuarterLabel"]).agg({"SalesAmount": "sum"})
        case "yearly":
            data_season = data.groupby("CalendarYear").agg({"SalesAmount": "sum"})

    return data_season[["SalesAmount"]]



def preprocess_features(X: pd.DataFrame, simple:bool = False) -> np.ndarray:
    print(Fore.BLUE + "\n[preprocessing] Preprocessing features..." + Style.RESET_ALL)

    preprocessor = SimplePreprocessor() if simple else SalesNinjaPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    if simple:
        X_processed = pd.DataFrame(X_processed)
        X_processed.columns = X.columns
        print("\n[preprocessing] X processed (simple), with shape ", X_processed.shape)
        return X_processed

    print("\n[preprocessing] X processed, with shape ", X_processed.shape)
    return X_processed



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
