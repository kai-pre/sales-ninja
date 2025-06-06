### Imports
import pandas as pd
import numpy as np
from os import getcwd, path



### Class definition

class SalesNinja():
    def __init__(self):
        pass

    def get_ml_data(self, ratio = 0.2):
        """
        Fetch raw data and merge it for machine learning use
        """
        # Load less data for now to try and avoid crashes
        skipratio = (1-ratio) # elements to skip

        skipsize = int(3406088 * skipratio)
        skipindices = np.random.choice(np.arange(1,3406088), (skipsize), replace = False)
        skipindices

        data = pd.read_csv(path.join(getcwd(), "raw_data", "FactSales.csv"), header = 0, skiprows = skipindices).drop(['CurrencyKey'], axis=1)
        #data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            path.join(getcwd(), "raw_data", "DimPromotion.csv"),
            usecols=["DiscountPercent", "PromotionKey"],
            ), on="PromotionKey", how="left",).merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimDate.csv"),
                usecols=[
                    "DateKey",
                    "IsHoliday",
                    "IsWorkDay",
                    "CalendarWeekLabel",
                    "CalendarYear",
                    "MonthNumber",
                    "CalendarQuarterLabel",
                    "CalendarDayOfWeekNumber"],
            ), on="DateKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimProduct.csv"),
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
                path.join(getcwd(), "raw_data", "DimProductSubcategory.csv"),
                usecols=["ProductSubcategoryKey", "ProductCategoryKey"],
            ), on="ProductSubcategoryKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimProductCategory.csv"),
                usecols=["ProductCategoryKey"],
            ), on="ProductCategoryKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimStore.csv"),
                usecols=[
                    "StoreKey",
                    "GeographyKey",
                    "StoreType",
                    "EmployeeCount",
                    "SellingAreaSize"],
            ), on="StoreKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimGeography.csv"),
                usecols=[
                    "GeographyKey",
                    "GeographyType",
                    "ContinentName",
                    "CityName",
                    "StateProvinceName",
                    "RegionCountryName",
                ],
            ), on="GeographyKey", how="left")

        return data

    def get_dashboard_data(self, ratio = 0.2):
        """
        Fetch raw data and merge it for dashboard use
        """
        # Load less data for now to try and avoid crashes
        skipratio = (1-ratio) # elements to skip

        skipsize = int(3406088 * skipratio)
        skipindices = np.random.choice(np.arange(1,3406088), (skipsize), replace = False)
        skipindices

        data = pd.read_csv(path.join(getcwd(), "raw_data", "FactSales.csv"), header = 0, skiprows = skipindices).drop(['CurrencyKey'], axis=1)
        #data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            path.join(getcwd(), "raw_data", "DimChannel.csv"), usecols=["ChannelKey", "ChannelName"]),
            left_on="channelKey",right_on="ChannelKey",how="left").rename(
                columns={"channelKey": "ChannelKey"}).merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimPromotion.csv"),
                usecols=["PromotionKey", "PromotionName", "PromotionType"]),
            on="PromotionKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimDate.csv"),
                usecols=[
                    "DateKey",
                    "CalendarYear",
                    "CalendarMonthLabel",
                    "MonthNumber",
                    "CalendarQuarterLabel",
                    "CalendarDayOfWeekNumber",
                    "CalendarDayOfWeekLabel"]),
            on="DateKey", how="left").merge(pd.read_csv(
            path.join(getcwd(), "raw_data", "DimProduct.csv"),
            usecols=["ProductKey", "ProductName", "ProductSubcategoryKey"]),
            on="ProductKey", how="left").merge(pd.read_csv(
            path.join(getcwd(), "raw_data", "DimProductSubcategory.csv"),
            usecols=[
                "ProductSubcategoryKey",
                "ProductSubcategoryName",
                "ProductCategoryKey"]),
            on="ProductSubcategoryKey", how = "left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimProductCategory.csv"),
                usecols=["ProductCategoryKey", "ProductCategoryName"]),
            on="ProductCategoryKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimStore.csv"),
                usecols=["StoreKey", "GeographyKey", "StoreType", "StoreName"]),
            on="StoreKey", how="left").merge(pd.read_csv(
                path.join(getcwd(), "raw_data", "DimGeography.csv"),
                usecols=["GeographyType",
                    "ContinentName",
                    "CityName",
                    "StateProvinceName",
                    "RegionCountryName",]),
            on="GeographyKey", how="left")

        return data

    def make_ml_csv(self, filename = "data_ml_merged.csv"):
        """
        Fetch raw data and create merged csv file for machine learning
        """
        filepath = path.join(getcwd(), "merged_data", filename)
        self.get_ml_data().to_csv(filepath)
        print(f"Data saved to '{filename}'")

    def make_dashboard_csv(self, filename = "data_dashboard_merged.csv"):
        """
        Fetch raw data and create merged csv file for dashboard
        """
        filepath = path.join(getcwd(), "merged_data", filename)
        self.get_dashboard_data().to_csv(filepath)
        print(f"Data saved to '{filename}'")

    def get_custom_data(self, *args):
        """
        Create custom dataframe from user input
        """
        pass
