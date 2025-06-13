### Imports
import pandas as pd
import numpy as np
import glob
import os
from os import getcwd, path

from google.cloud import bigquery, storage
from colorama import Style, Fore
from pathlib import Path

from salesninja.params import *

### Class definition

class SalesNinja():
    def __init__(self):
        pass

    def make_ml_data(self, ratio = DATA_SIZE):
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
                         path.join(LOCAL_DATA_PATH, "merged", f"data_ml_merged_{int(ratio*100)}.csv"))
        self.load_data_to_bq(
            data,
            f"data_ml_merged_{int(ratio*100)}",
            truncate = True
        )

        return data


    def make_db_data(self, ratio = DATA_SIZE):
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
                         path.join(LOCAL_DATA_PATH, "merged", f"data_db_merged_{int(ratio*100)}.csv"))
        self.load_data_to_bq(
            data,
            f"data_db_merged_{int(ratio*100)}",
            truncate = True
        )

        return data


    def get_ml_data(self, ratio = DATA_SIZE, min_date = "2007-01-01", max_date = "2009-12-31"):
        """
        Fetch merged data for machine learning use
        """
        local_data_directory = path.join(LOCAL_DATA_PATH, "merged")

        if Path(path.join(local_data_directory, f"data_ml_merged_{int(ratio*100)}.csv")).is_file():
            print("[SalesNinja] Loading local merged ML data ...")
            data = pd.read_csv(path.join(local_data_directory, f"data_ml_merged_{int(ratio*100)}.csv"))
        else:
            print(f"[SalesNinja] No local merged ML data with ratio {ratio} found, merging raw files ...")
            data = self.make_ml_data(ratio = ratio)

        data = data[(data['DateKey'] >= min_date) & (data['DateKey'] <= max_date)]

        return data


    def get_db_data(self, ratio = DATA_SIZE, min_date = "2007-01-01", max_date = "2009-12-31"):
        """
        Fetch merged data for dashboard use
        """
        local_data_directory = path.join(LOCAL_DATA_PATH, "merged")

        if Path(path.join(local_data_directory, f"data_db_merged_{int(ratio*100)}.csv")).is_file():
            print("[SalesNinja] Loading local merged dashboard data ...")
            data = pd.read_csv(path.join(local_data_directory, f"data_db_merged_{int(ratio*100)}.csv"))
        else:
            print(f"[SalesNinja] No local merged dashboard data with ratio {ratio} found, merging raw files ...")
            data = self.make_db_data(ratio = ratio)

        data = data[(data['DateKey'] >= min_date) & (data['DateKey'] <= max_date)]

        return data


    def save_as_csv(self, mergeddata, filename = "unknown/data_merged.csv"):
        """
        Save dataframe as CSV
        """
        #local_filename = Path(path.join(LOCAL_DATA_PATH, filename))
        if not path.exists(path.dirname(filename)):
            print(f"- Path does not exist, will create {path.dirname(filename)}")
            os.makedirs(path.dirname(filename))
        else:
            print(f"- Path {path.dirname(filename)} exists, will do nothing ...")

        mergeddata.to_csv(filename, header=True, index=False)
        print(f"- Data saved to '{filename}'")


    def get_custom_data(self, *args):
        """
        Create custom dataframe from user input
        """
        pass


    def get_data_with_cache(
            self,
            query:str,
            cache_path:Path,
            data_has_header=True
        ) -> pd.DataFrame:
        """
        Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
        Store at `cache_path` if retrieved from BigQuery for future use
        """

        if cache_path.is_file():
            print(Fore.BLUE + "\n[ML] Load data from local CSV..." + Style.RESET_ALL)
            df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
            #df = pd.read_csv(cache_path, header = 0)
        else:
            print(Fore.BLUE + "\n[ML] Load data from BigQuery server..." + Style.RESET_ALL)
            client = bigquery.Client(project=GCP_SALESNINJA)
            query_job = client.query(query)
            result = query_job.result()
            df = result.to_dataframe()

            # Store as CSV if the BQ query returned at least one valid line
            if df.shape[0] > 1:
                if not path.exists(path.dirname(cache_path)):
                    os.makedirs(path.dirname(cache_path))
                df.to_csv(cache_path, header=data_has_header, index=False)

        print(f"[ML] Data loaded, with shape {df.shape}")

        return df


    def get_data_with_cache_and_filter(
            self,
            query:str,
            cache_path:Path,
            min_date:str,
            max_date:str,
            data_has_header=True
        ) -> pd.DataFrame:
        """
        Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
        Store at `cache_path` if retrieved from BigQuery for future use
        """

        if cache_path.is_file():
            print(Fore.BLUE + "\n[ML] Load data from local CSV..." + Style.RESET_ALL)
            df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
            #df = pd.read_csv(cache_path, header = 0)
        else:
            print(Fore.BLUE + "\n[ML] Load data from BigQuery server..." + Style.RESET_ALL)
            client = bigquery.Client(project=GCP_SALESNINJA)
            query_job = client.query(query)
            result = query_job.result()
            df = result.to_dataframe()

            # Store as CSV if the BQ query returned at least one valid line
            if df.shape[0] > 1:
                if not path.exists(path.dirname(cache_path)):
                    os.makedirs(path.dirname(cache_path))
                df.to_csv(cache_path, header=data_has_header, index=False)

        df = df[(df['DateKey'] >= min_date) & (df['DateKey'] <= max_date)]
        print(f"[ML] Data loaded, with shape {df.shape}")

        return df


    def load_data_to_bq(
            self,
            data: pd.DataFrame,
            table: str,
            truncate: bool
        ) -> None:
        """
        - Save the DataFrame to BigQuery
        - Empty the table beforehand if `truncate` is True, append otherwise
        """

        assert isinstance(data, pd.DataFrame)
        full_table_name = f"{GCP_SALESNINJA}.{BQ_DATASET}.{table}"
        print(Fore.BLUE + f"\n[ML] Saving data to BigQuery @ {full_table_name} ...:" + Style.RESET_ALL)

        # Load data onto full_table_name

        # TODO: simplify this solution if possible, but students may very well choose another way to do it
        # We don't test directly against their own BQ tables, but only the result of their query
        # data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

        client = bigquery.Client()

        # Define write mode and schema
        write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
        job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

        print(f"\n[ML] {'Writing' if truncate else 'Appending'} {full_table_name} ({data.shape[0]} rows)")

        # Load data
        job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
        result = job.result()  # wait for the job to complete

        print(f"[ML] Data saved to bigquery, with shape {data.shape}")

if __name__ == "__main__":
    SalesNinja().get_ml_data(ratio = 0.2)


# ML data is:
"""
SalesKey,DateKey,channelKey,StoreKey,ProductKey,PromotionKey,UnitCost,UnitPrice,
SalesQuantity,ReturnQuantity,ReturnAmount,DiscountQuantity,DiscountAmount,TotalCost,
SalesAmount,DiscountPercent,CalendarYear,CalendarQuarterLabel,CalendarWeekLabel,
IsWorkDay,MonthNumber,CalendarDayOfWeekNumber,ProductSubcategoryKey,BrandName,
ClassID,StyleID,ColorID,Weight,WeightUnitMeasureID,StockTypeID,ProductCategoryKey,
GeographyKey,StoreType,EmployeeCount,SellingAreaSize,GeographyType,ContinentName,
CityName,StateProvinceName,RegionCountryName
"""
