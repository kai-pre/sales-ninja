### Imports
##### loads a GPU-accelerated version of pandas, disable if this creates any problems!!
"""
try:
    import cudf.pandas
    cudf.pandas.install()
except:
    pass
"""
#####
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import sys
from os import path
#from datetime import timedelta

#from colorama import Fore, Style
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.cag import FixedCombinations, OneHotEncoding

from salesninja.params import *


local_synthmodel_path = path.join(LOCAL_REGISTRY_PATH, "synthmodels")



class SalesNinjaSynthesis():
    def __init__(self):
        pass

    def initialize(self, dataframe):
        self.metadata = Metadata.detect_from_dataframe(
            data=dataframe,
            table_name='SalesData'
        )

        # new columns:
        # UnitCost,UnitPrice,ReturnQuantity,ReturnAmount,DiscountQuantity,
        # DiscountAmount,DiscountPercent,Weight_grams,IsWorkDay_WeekEnd,
        # IsWorkDay_WorkDay,BrandName_A_Datum,BrandName_Adventure_Works,
        # BrandName_Contoso,BrandName_Fabrikam,BrandName_Litware,BrandName_NA,
        # BrandName_Northwind_Traders,BrandName_Proseware,BrandName_Southridge_Video,
        # BrandName_Tailspin_Toys,BrandName_The_Phone_Company,
        # BrandName_Wide_World_Importers,StoreType_Catalog,StoreType_Online,
        # StoreType_Reseller,StoreType_Store,GeographyType_City,
        # RegionCountryName_Armenia,RegionCountryName_Australia,
        # RegionCountryName_Bhutan,RegionCountryName_Canada,
        # RegionCountryName_China,RegionCountryName_Denmark,
        # RegionCountryName_France,RegionCountryName_Germany,
        # RegionCountryName_Greece,RegionCountryName_India,
        # RegionCountryName_Iran,RegionCountryName_Ireland,
        # RegionCountryName_Italy,RegionCountryName_Japan,
        # RegionCountryName_Kyrgyzstan,RegionCountryName_Malta,
        # RegionCountryName_Pakistan,RegionCountryName_Poland,
        # RegionCountryName_Portugal,RegionCountryName_Romania,
        # RegionCountryName_Russia,RegionCountryName_Singapore,
        # RegionCountryName_Slovenia,RegionCountryName_South_Korea,
        # RegionCountryName_Spain,RegionCountryName_Sweden,
        # RegionCountryName_Switzerland,RegionCountryName_Syria,
        # RegionCountryName_Taiwan,RegionCountryName_Thailand,
        # RegionCountryName_Turkmenistan,RegionCountryName_United_Kingdom,
        # RegionCountryName_United_States,RegionCountryName_the_Netherlands,
        # SalesQuantity,sin_MonthNumber,cos_MonthNumber,
        # sin_CalendarDayOfWeekNumber,cos_CalendarDayOfWeekNumber,
        # sin_CalendarQuarterLabel,cos_CalendarQuarterLabel,SalesAmount

        # if DateKey or CalendarYear are still columns in dataframe, drop them
        if "SalesKey" in dataframe.columns:
            dataframe = dataframe.drop('SalesKey', axis = 1)
        if "DateKey" in dataframe.columns:
            dataframe = dataframe.drop('DateKey', axis = 1)
        if "CalendarYear" in dataframe.columns:
            dataframe = dataframe.drop('CalendarYear', axis = 1)

        self.metadata.update_columns(
            column_names=[
                "GeographyType_City"
            ],
            sdtype="categorical"
        )

        self.metadata.update_columns(
            column_names=[
                "DiscountAmount",
                "DiscountPercent",
                "DiscountQuantity"
            ],
            sdtype="categorical"
        )

        """
        self.metadata.update_column(
            column_name = "DateKey",
            sdtype = "datetime",
            datetime_format = "%Y-%m-%d"
        )

        self.metadata.update_column(
            column_name = "CalendarYear",
            sdtype = "datetime",
            datetime_format = "%Y"
        )

        self.metadata.update_column(
            column_name = "MonthNumber",
            sdtype = "datetime",
            datetime_format = "%m"
        )
        """

        self.constraints = [
            FixedCombinations(
                column_names=['DiscountAmount', 'DiscountPercent', 'DiscountQuantity']
            ),
            OneHotEncoding(
                column_names=["IsWorkDay_WeekEnd", "IsWorkDay_WorkDay"]
            )
        ]

        self.model = GaussianCopulaSynthesizer(
            self.metadata,
            #enforce_min_max_values=True, # seems to freeze in sampling step for some reason
            enforce_min_max_values=False,
            enforce_rounding=True,
            numerical_distributions=dict(
                UnitCost = "truncnorm",
                UnitPrice = "truncnorm",
                SalesQuantity = "gamma",
                ReturnAmount = "truncnorm",
                SalesAmount = "truncnorm",
            ),
            default_distribution='beta'
        )

        self.trainlen = dataframe.shape[0]
        self.model.add_constraints(constraints=self.constraints)

        return self


    def show_metadata(self, forceprint = False):
        if forceprint:
            print(self.metadata.visualize())
        else:
            self.metadata.visualize()


    def create_facts(self, min_date, max_date):
        """
        Creates known facts about the provided date range. Facts are:
        IsWorkDay_WeekEnd, IsWorkDay_WorkDay, sin_MonthNumber,cos_MonthNumber,
        sin_CalendarDayOfWeekNumber,cos_CalendarDayOfWeekNumber,
        sin_CalendarQuarterLabel,cos_CalendarQuarterLabel

        DateKey and CalendarYear are technically facts, but not needed for
        data generation.
        """

        daterange = pd.date_range(pd.to_datetime(min_date), pd.to_datetime(max_date), freq='d').to_series()
        calendardayofweeknumber = daterange.dt.dayofweek.values + 1
        #calendaryear = daterange.dt.year.values
        monthnumber = daterange.dt.month.values
        calendarquarterlabel = ((monthnumber-1) // 3) + 1
        #calendarweeklabel = np.char.add("Week ", np.char.mod("%d", daterange.dt.isocalendar().week))
        isworkday = np.less_equal(calendardayofweeknumber, 5)# <= 5

        newdata = pd.DataFrame({
            "DateKey": daterange,
            "sin_CalendarQuarterLabel": np.sin(2*np.pi*calendarquarterlabel/4),
            "cos_CalendarQuarterLabel": np.cos(2*np.pi*calendarquarterlabel/4),
            "IsWorkDay_WeekEnd": np.where(isworkday, 0, 1),
            "IsWorkDay_WorkDay": np.where(isworkday, 1, 0),
            "sin_MonthNumber": np.sin(2*np.pi*monthnumber/12),
            "cos_MonthNumber": np.cos(2*np.pi*monthnumber/12),
            "sin_CalendarDayOfWeekNumber": np.sin(2*np.pi*calendardayofweeknumber/7),
            "cos_CalendarDayOfWeekNumber": np.cos(2*np.pi*calendardayofweeknumber/7)
            })
        newdata.reset_index(drop = True, inplace = True)

        self.factslen = len(daterange)

        return newdata


    def fit(self, data):
        # data["GeographyType_City"] = data["GeographyType_City"].astype(int)
        # data[["DiscountAmount",
        #       "DiscountPercent",
        #       "DiscountQuantity"]] = data[["DiscountAmount",
        #                                    "DiscountPercent",
        #                                    "DiscountQuantity"]].astype(object)
        self.model.fit(data)
        """
        self.model.save(
            filepath=path.join(local_synthmodel_path, 'CPSynthesizer.pkl')
        )
        """
        return self


    def sample(self, rows = None, factor = 1):
        if rows is None:
            rows = self.trainlen
        """
        Create synthetic data for the the defined date range, based on existing data
        """
        return self.model.sample(num_rows = rows * factor)


    def sample_with_facts(self, factsdata, batch_size = 1):
        """
        Create synthetic data for the the defined date range, based on existing data
        and using known facts
        """
        ignorable_date_columns = factsdata[["DateKey"]]
        facts_without_date = factsdata.drop(columns = ["DateKey"])

        synthdata_without_date = self.model.sample_remaining_columns(
            known_columns=facts_without_date,
            batch_size = batch_size,
            max_tries_per_batch=100)

        #return pd.concat([ignorable_date_columns, synthdata_without_date], axis = 1)
        return synthdata_without_date


    def load(self):
        """
        If synthetic data is saved locally or in GCS, load it, otherwise create it
        """
        loadpath = path.join(local_synthmodel_path, 'CPSynthesizer.pkl')
        if loadpath.is_file():
            print(f"\n[Synthesis] GPSynthesizer found, loading ...")
            self.model = GaussianCopulaSynthesizer.load(
                filepath=path.join(local_synthmodel_path, 'CPSynthesizer.pkl')
            )
        else:
            print(f"\n[Synthesis] GPSynthesizer not found, creating new one ...")
            self.model = self.initialize()
        return self.model


    def evaluate(self, actual_data, synth_data, withplot = False):
        from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
        from sdv.evaluation.single_table import get_column_plot

        # 1. perform basic validity checks
        diagnostic = run_diagnostic(actual_data, synth_data, self.metadata)

        print("---------------------------------------------------")
        # 2. measure the statistical similarity
        quality_report = evaluate_quality(actual_data, synth_data, self.metadata)

        # 3. plot the data
        if withplot:
            fig = get_column_plot(
                real_data=actual_data,
                synthetic_data=synth_data,
                metadata=self.metadata,
                column_name='SalesAmount'
            )
            fig.show()



if __name__ == '__main__':
    from salesninja.data import SalesNinja
    from salesninja.ml.registry import load_model
    from salesninja.ml.preprocessing import preprocess_features, seasonalize_data, seasonalize_y
    from sklearn.metrics import mean_absolute_error, r2_score

    print("----- Getting test data -----")
    testdata = SalesNinja().get_ml_data(ratio = 0.1, min_date = "2007-01-01", max_date = "2009-05-31")
    y_baseline = seasonalize_y(testdata).mean().tolist() * 30
    testdata = preprocess_features(testdata)
    testdata = seasonalize_data(testdata)
    print("----- Getting val data -----")
    valdata = SalesNinja().get_ml_data(ratio = 0.1, min_date = "2009-06-01", max_date = "2009-06-30")
    y_true = seasonalize_y(valdata)
    valdata = preprocess_features(valdata)
    valdata = seasonalize_data(valdata)
    print("----- Creating synthesis model -----")
    testsynth = SalesNinjaSynthesis()
    testsynth = testsynth.initialize(testdata)
    print("----- Showing Metadata -----")
    testsynth.show_metadata()
    print("----- Fitting synthesis model -----")
    testsynth = testsynth.fit(testdata)
    print("----- Creating facts -----")
    testfacts = testsynth.create_facts("2009-06-01", "2009-06-30")
    print("----- Creating synth data -----")
    testsynthdata = testsynth.sample_with_facts(testfacts)
    print("----- Starting evaluation -----")
    testsynth.evaluate(valdata, testsynthdata)# takes a while (20 seconds)
    print("----- Starting prediction -----")
    testsynthdata = testsynthdata.drop("SalesAmount", axis = 1)
    testsynthdata = testsynthdata.drop("DateKey", axis = 1)

    model = load_model()
    assert model is not None

    y_pred = model.predict(testsynthdata)
    print("----- Evaluating prediction -----")
    baselinemae = mean_absolute_error(y_true, y_baseline)
    baseliner2 = r2_score(y_true, y_baseline)
    print(f"----- Baseline results: synth prediction MAE is {baselinemae}, synth prediction R2 is {baseliner2} ! -----")
    testmae = mean_absolute_error(y_true, y_pred)
    testr2 = r2_score(y_true, y_pred)
    print(f"----- Final results: synth prediction MAE is {testmae}, synth prediction R2 is {testr2} ! -----")
