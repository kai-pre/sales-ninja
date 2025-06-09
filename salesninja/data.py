### Imports
import pandas as pd
import numpy as np
import glob
from os import getcwd, path


from salesninja.params import *

### Class definition

class SalesNinja():
    def __init__(self):
        pass

    def make_ml_data(self, ratio = 0.2):
        """
        Fetch raw data and merge it for ML use
        """
        local_rawdata_directory = path.join(LOCAL_DATA_PATH, "raw")

        # Load less data for now to try and avoid crashes
        skipratio = (1-ratio) # elements to skip
        skipsize = int(NUMBER_OF_ROWS * skipratio)
        skipindices = np.random.choice(np.arange(1,NUMBER_OF_ROWS), (skipsize), replace = False)

        data = pd.read_csv(path.join(local_rawdata_directory, "FactSales.csv"),
                           header = 0, skiprows = skipindices).drop(['CurrencyKey'], axis=1)
        #data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            path.join(local_rawdata_directory, "DimPromotion.csv"),
            usecols=["DiscountPercent", "PromotionKey"],
            ), on="PromotionKey", how="left",).merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimDate.csv"),
                usecols=[
                    "DateKey",
                    "IsWorkDay",
                    "CalendarWeekLabel",
                    "CalendarYear",
                    "MonthNumber",
                    "CalendarQuarterLabel",
                    "CalendarDayOfWeekNumber"],
            ), on="DateKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimProduct.csv"),
                usecols=[
                    "ProductKey",
                    "ProductSubcategoryKey",
                    "BrandName",
                    "ClassID",
                    "StyleID",
                    "ColorID",
                    "Weight",
                    "WeightUnitMeasureID",
                    "StockTypeID"],
            ), on="ProductKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimProductSubcategory.csv"),
                usecols=["ProductSubcategoryKey", "ProductCategoryKey"],
            ), on="ProductSubcategoryKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimProductCategory.csv"),
                usecols=["ProductCategoryKey"],
            ), on="ProductCategoryKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimStore.csv"),
                usecols=[
                    "StoreKey",
                    "GeographyKey",
                    "StoreType",
                    "EmployeeCount",
                    "SellingAreaSize"],
            ), on="StoreKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimGeography.csv"),
                usecols=[
                    "GeographyKey",
                    "GeographyType",
                    "ContinentName",
                    "CityName",
                    "StateProvinceName",
                    "RegionCountryName",
                ],
            ), on="GeographyKey", how="left")

        self.save_as_csv(data,
                         path.join(LOCAL_DATA_PATH, "ml_merged", f"ratio{ratio}", "data_ml_merged.csv"))

        return data


    def make_db_data(self, ratio = 0.2):
        """
        Fetch raw data and merge it for dashboard use
        """
        local_rawdata_directory = path.join(LOCAL_DATA_PATH, "raw")

        # Load less data for now to try and avoid crashes
        skipratio = (1-ratio) # elements to skip

        skipsize = int(NUMBER_OF_ROWS * skipratio)
        skipindices = np.random.choice(np.arange(1,NUMBER_OF_ROWS), (skipsize), replace = False)
        skipindices

        data = pd.read_csv(path.join(local_rawdata_directory, "FactSales.csv"),
                           header = 0, skiprows = skipindices).drop(['CurrencyKey'], axis=1)
        #data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            path.join(path.join(local_rawdata_directory, "DimChannel.csv"),
            usecols=["ChannelKey", "ChannelName"]),
            left_on="channelKey",right_on="ChannelKey",how="left")).rename(
                columns={"channelKey": "ChannelKey"}).merge(pd.read_csv(
                path.join(path.join(local_rawdata_directory, "DimPromotion.csv"),
                usecols=["PromotionKey", "PromotionName", "PromotionType"]),
            on="PromotionKey", how="left").merge(path.join(local_rawdata_directory, "DimDate.csv"),
                usecols=[
                    "DateKey",
                    "CalendarYear",
                    "CalendarMonthLabel",
                    "MonthNumber",
                    "CalendarQuarterLabel",
                    "CalendarDayOfWeekNumber",
                    "CalendarDayOfWeekLabel"]),
            on="DateKey", how="left").merge(pd.read_csv(
            path.join(local_rawdata_directory, "DimProduct.csv"),
            usecols=["ProductKey", "ProductName", "ProductSubcategoryKey"]),
            on="ProductKey", how="left").merge(pd.read_csv(
            path.join(local_rawdata_directory, "DimProductSubcategory.csv"),
            usecols=[
                "ProductSubcategoryKey",
                "ProductSubcategoryName",
                "ProductCategoryKey"]),
            on="ProductSubcategoryKey", how = "left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimProductCategory.csv"),
                usecols=["ProductCategoryKey", "ProductCategoryName"]),
            on="ProductCategoryKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimStore.csv"),
                usecols=["StoreKey", "GeographyKey", "StoreType", "StoreName"]),
            on="StoreKey", how="left").merge(pd.read_csv(
                path.join(local_rawdata_directory, "DimGeography.csv"),
                usecols=["GeographyType",
                    "ContinentName",
                    "CityName",
                    "StateProvinceName",
                    "RegionCountryName",]),
            on="GeographyKey", how="left")

        self.save_as_csv(data,
                         path.join(LOCAL_DATA_PATH, "db_merged", f"ratio{ratio}", "data_db_merged.csv"))

        return data


    def get_ml_data(self, ratio = 0.2, starttime = "2007-01-01", endtime = "2009-12-31"):
        """
        Fetch merged data for machine learning use
        """
        local_data_directory = path.join(LOCAL_DATA_PATH, "ml_merged", f"ratio{ratio}")

        try:
            data = pd.read_csv(path.join(local_data_directory, "data_ml_merged.csv"))
            print("[SalesNinja] Successfully loaded local merged ML data")
        except:
            print("[SalesNinja] No local merged ML data found, merging raw files...")
            data = self.make_ml_data(ratio = ratio)

        return data


    def get_db_data(self, ratio = 0.2, starttime = "2007-01-01", endtime = "2009-12-31"):
        """
        Fetch merged data for dashboard use
        """
        local_data_directory = path.join(LOCAL_DATA_PATH, "db_merged", f"ratio{ratio}")

        try:
            data = pd.read_csv(path.join(local_data_directory, "data_db_merged.csv"))
            print("[SalesNinja] Successfully loaded local merged dashboard data")
        except:
            print("[SalesNinja] No local merged dashboard data found, merging raw files...")
            data = self.make_db_data(ratio = ratio)

        return data


    def save_as_csv(self, mergeddata, filename = "unknown/data_merged.csv"):
        """
        Save dataframe as CSV
        """
        local_filename = path.join(LOCAL_DATA_PATH, filename)
        mergeddata.to_csv(local_filename)
        print(f"- Data saved to '{local_filename}'")


    def get_custom_data(self, *args):
        """
        Create custom dataframe from user input
        """
        pass
