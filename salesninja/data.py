### Imports
import pandas as pd
import os



### Class definition

class SalesNinja():
    def get_ml_data(self):
        """
        Fetch raw data and merge it for machine learning use
        """
        data = pd.read_csv("../raw_data/FactSales.csv").drop(['CurrencyKey'], axis=1)
        data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            "../raw_data/DimPromotion.csv",
            usecols=["DiscountPercent", "PromotionKey"],
            ), on="PromotionKey", how="left",).merge(pd.read_csv(
                "../raw_data/DimDate.csv",
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
                "../raw_data/DimProductSubcategory.csv",
                usecols=["ProductSubcategoryKey", "ProductCategoryKey"],
            ), on="ProductSubcategoryKey", how="left").merge(pd.read_csv(
                "../raw_data/DimProductCategory.csv",
                usecols=["ProductCategoryKey"],
            ), on="ProductCategoryKey", how="left").merge(pd.read_csv(
                "../raw_data/DimProduct.csv",
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
                "../raw_data/DimStore.csv",
                usecols=[
                    "StoreKey",
                    "GeographyKey",
                    "StoreType",
                    "EmployeeCount",
                    "SellingAreaSize"],
            ), on="StoreKey", how="left").merge(pd.read_csv(
                "../raw_data/DimGeography.csv",
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

    def get_dashboard_data(self):
        """
        Fetch raw data and merge it for dashboard use
        """
        data = pd.read_csv("../raw_data/FactSales.csv").drop(['CurrencyKey'], axis=1)
        data.set_index(['SalesKey'], inplace=True)

        data = data.merge(pd.read_csv(
            "../raw_data/DimChannel.csv", usecols=["ChannelKey", "ChannelName"]),
            on="channelKey",how="left").merge(pd.read_csv(
                "../raw_data/DimPromotion.csv",
                usecols=["PromotionKey", "PromotionName", "PromotionType"]),
            on="PromotionKey", how="left").merge(pd.read_csv(
                "../raw_data/DimDate.csv",
                usecols=[
                    "DateKey",
                    "CalendarYear",
                    "CalendarMonthLabel",
                    "MonthNumber",
                    "CalendarQuarterLabel",
                    "CalendarDayOfWeekNumber",
                    "CalendarDayOfWeekLabel"]),
            on="DateKey", how="left").merge(pd.read_csv(
                "../raw_data/DimProductSubcategory.csv",
                usecols=[
                    "ProductSubcategoryKey",
                    "ProductSubcategoryName",
                    "ProductCategoryKey"]),
            on="ProductSubcategoryKey", how = "left").merge(pd.read_csv(
                    "../raw_data/DimProductCategory.csv",
                    usecols=["ProductCategoryKey", "ProductCategoryName"]),
            on="ProductCategoryKey", how="left").merge(pd.read_csv(
                "../raw_data/DimProduct.csv",
                usecols=["ProductKey", "ProductName", "ProductSubcategoryKey"]),
            on="ProductKey", how="left").merge(pd.read_csv(
                "../raw_data/DimStore.csv",
                usecols=["StoreKey", "GeographyKey", "StoreType", "StoreName"]),
            on="StoreKey", how="left").merge(pd.read_csv(
                "../raw_data/DimGeography.csv",
                usecols=["GeographyKey", "ContinentName"]),
            on="GeographyKey", how="left")

        return data

    def make_ml_csv(self, filename = "data_ml_merged.csv"):
        """
        Fetch raw data and create merged csv file for machine learning
        """
        self.get_ml_data().to_csv(filename)
        print(f"Data saved to '{filename}'")

    def make_dashboard_csv(self, filename = "data_dashboard_merged.csv"):
        """
        Fetch raw data and create merged csv file for dashboard
        """
        self.get_dashboard_data().to_csv(filename)
        print(f"Data saved to '{filename}'")

    def get_custom_data(self, *args):
        """
        Create custom dataframe from user input
        """
        pass
