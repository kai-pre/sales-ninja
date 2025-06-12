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

        # if DateKey or CalendarYear are still columns in dataframe, drop them
        if "SalesKey" in dataframe.columns:
            dataframe = dataframe.drop('SalesKey', axis = 1)
        if "DateKey" in dataframe.columns:
            dataframe = dataframe.drop('DateKey', axis = 1)
        if "CalendarYear" in dataframe.columns:
            dataframe = dataframe.drop('CalendarYear', axis = 1)

        self.metadata.update_columns(
            column_names = [
                "channelKey", "StoreKey", "ProductKey", "PromotionKey",
                "CalendarQuarterLabel", "CalendarWeekLabel",
                "IsWorkDay","CalendarDayOfWeekNumber", "ProductSubcategoryKey",
                "BrandName", "ClassID", "StyleID", "ColorID", "WeightUnitMeasureID",
                "StockTypeID", "ProductCategoryKey", "GeographyKey", "StoreType",
                "GeographyType", "ContinentName", "CityName", "StateProvinceName",
                "RegionCountryName", "SellingAreaSize", "DiscountAmount",
                "DiscountPercent", "DiscountQuantity", "ReturnQuantity", "Weight",
                ], sdtype="categorical"
        )

        self.metadata.update_columns(
            column_names = [
                None
                ], sdtype="numerical",
                computer_representation = "UInt32"
        )

        self.metadata.update_columns(
            column_names = [
                None
                ], sdtype="numerical",
                computer_representation = "Float"
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
        """

        self.metadata.update_column(
            column_name = "MonthNumber",
            sdtype = "datetime",
            datetime_format = "%m"
        )


        self.constraints = [
            FixedCombinations(
                column_names=['CityName', 'ContinentName', 'StateProvinceName',
                              'GeographyKey', 'GeographyType', 'RegionCountryName']
            ),
            FixedCombinations(
                column_names=['StoreKey', 'StoreType', 'SellingAreaSize']
            ),
            FixedCombinations(
                column_names=['DiscountAmount', 'DiscountPercent', 'DiscountQuantity']
            ),
            FixedCombinations(
                column_names=['Weight', 'WeightUnitMeasureID']
            )
        ]

        self.model = GaussianCopulaSynthesizer(
            self.metadata,
            #enforce_min_max_values=True,
            enforce_min_max_values=False,
            enforce_rounding=False,
            numerical_distributions=dict(
                UnitCost = "truncnorm",
                UnitPrice = "truncnorm",
                SalesQuantity = "gamma",
                ReturnAmount = "truncnorm",
                TotalCost = "truncnorm",
                SalesAmount = "truncnorm",
                EmployeeCount = "beta",
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
        CalendarQuarterLabel, CalendarWeekLabel, IsWorkDay, MonthNumber,
        CalendarDayOfWeekNumber. DateKey and CalendarYear are technically facts,
        but not needed for data generation
        """
        ### TO DO: import actual data if range lies in 2007-2009?
        ### TO DO: maybe drop DateKey after

        daterange = pd.date_range(pd.to_datetime(min_date), pd.to_datetime(max_date), freq='d').to_series()
        calendardayofweeknumber = daterange.dt.dayofweek.values + 1
        calendaryear = daterange.dt.year.values
        monthnumber = daterange.dt.month.values
        calendarquarterlabel = np.char.add("Q", np.char.mod("%d", (((monthnumber-1) // 3) + 1)))
        calendarweeklabel = np.char.add("Week ", np.char.mod("%d", daterange.dt.isocalendar().week))
        isworkday = np.less_equal(calendardayofweeknumber, 5)# <= 5
        isworkday = np.where(isworkday, "WorkDay", "WeekEnd")

        newdata = pd.DataFrame({
            "DateKey": daterange, # actually leave this out
            "CalendarYear": calendaryear, # actually leave this out
            "CalendarQuarterLabel": calendarquarterlabel,
            "CalendarWeekLabel": calendarweeklabel,
            "IsWorkDay": isworkday,
            "MonthNumber": monthnumber,
            "CalendarDayOfWeekNumber": calendardayofweeknumber
            })
        newdata.reset_index(drop = True, inplace = True)

        self.factslen = len(daterange)

        return newdata


    def fit(self, data):
        data[[
            "ProductSubcategoryKey", "ClassID", "StyleID",
            "ColorID", "Weight", "ProductCategoryKey",
            "StockTypeID"
            ]] = data[[
            "ProductSubcategoryKey", "ClassID", "StyleID",
            "ColorID", "Weight", "ProductCategoryKey",
            "StockTypeID"
            ]].astype(int)

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


    def sample_with_facts(self, factsdata, batch_size = 7):
        """
        Create synthetic data for the the defined date range, based on existing data
        and using known facts
        """
        ignorable_date_columns = factsdata[["DateKey", "CalendarYear"]]
        facts_without_date = factsdata.drop(columns = ["DateKey", "CalendarYear"])

        synthdata_without_date = self.model.sample_remaining_columns(
            known_columns=facts_without_date,
            batch_size = batch_size,
            max_tries_per_batch=500)

        return pd.concat([ignorable_date_columns, synthdata_without_date], axis = 1)


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
        #print(diagnostic)

        print("---------------------------------------------------")
        # 2. measure the statistical similarity
        quality_report = evaluate_quality(actual_data, synth_data, self.metadata)
        #print(quality_report)

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
    from salesninja.ml.preprocessing import preprocess_features, seasonalize_data
    from sklearn.metrics import mean_absolute_error, r2_score

    print("----- Getting test data -----")
    testdata = SalesNinja().get_ml_data(min_date = "2008-01-01", max_date = "2008-12-31")
    print("----- Getting val data -----")
    valdata = SalesNinja().get_ml_data(min_date = "2009-01-01", max_date = "2009-01-15")
    print("----- Creating synthesis model -----")
    testsynth = SalesNinjaSynthesis()
    testsynth = testsynth.initialize(testdata)
    print("----- Showing Metadata -----")
    testsynth.show_metadata()
    print("----- Fitting synthesis model -----")
    testsynth = testsynth.fit(testdata)
    print("----- Creating facts -----")
    testfacts = testsynth.create_facts("2009-01-01", "2009-01-15")
    print(testfacts)
    print("----- Creating synth data -----")
    testsynthdata = testsynth.sample_with_facts(testfacts)
    print(testsynthdata.to_string())

    ########### TO DO: NaNs appear for category values that are float. Set their dtypes to int32 specifically!!!

    """
    print("----- Starting evaluation -----")
    testsynth.evaluate(valdata, testsynthdata)


    print("----- Starting prediction -----")
    testsynthdata = testsynthdata.drop("SalesAmount", axis = 1)
    print(testsynthdata.to_string())

    model = load_model()
    assert model is not None

    testsynthdata_processed = preprocess_features(testsynthdata, simple = False)
    testsynthdata_processed = seasonalize_data(testsynthdata_processed)

    testsynthdata_processed = testsynthdata_processed.drop("DateKey", axis = 1)
    y_pred = model.predict(testsynthdata_processed)
    print("----- Evaluating prediction -----")
    y_true = valdata["SalesAmount"]
    testmae = mean_absolute_error(y_true, y_pred)
    testr2 = r2_score(y_true, y_pred)
    print(f"----- Final results: synth prediction MAE is {testmae}, synth prediction R2 is {testr2} ! -----")
    """
