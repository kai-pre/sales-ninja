### Imports
##### loads a GPU-accelerated version of pandas, disable if this creates any problems!!
try:
    import cudf.pandas
    cudf.pandas.install()
except:
    pass
#####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from os import path

from colorama import Fore, Style
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.cag import FixedCombinations, OneHotEncoding

from salesninja.params import *

local_synthmodel_path = path.join(LOCAL_DATA_PATH, "synthmodels")



class SalesNinjaSynthesis():
    def __init__(self):
        pass

    def initialize(self, dataframe):
        self.metadata = Metadata.detect_from_dataframe(
            data=dataframe,
            table_name='SalesData'
        )
        self.metadata.update_column(
            column_name='DateKey',
            sdtype='datetime',
            datetime_format='%Y-%m-%d'
        )

        self.constraints = [
            FixedCombinations(
                column_names=['CityName', 'ContinentName', 'StateProvinceName',
                              'GeographyKey', 'GeographyType_City',
                              'RegionCountryName_Armenia',
                              'RegionCountryName_Australia', 'RegionCountryName_Bhutan',
                                'RegionCountryName_Canada', 'RegionCountryName_China',
                                'RegionCountryName_Denmark', 'RegionCountryName_France',
                                'RegionCountryName_Germany', 'RegionCountryName_Greece',
                                'RegionCountryName_India', 'RegionCountryName_Iran',
                                'RegionCountryName_Ireland', 'RegionCountryName_Italy',
                                'RegionCountryName_Japan', 'RegionCountryName_Kyrgyzstan',
                                'RegionCountryName_Malta', 'RegionCountryName_Pakistan',
                                'RegionCountryName_Poland', 'RegionCountryName_Portugal',
                                'RegionCountryName_Romania', 'RegionCountryName_Russia',
                                'RegionCountryName_Singapore', 'RegionCountryName_Slovenia',
                                'RegionCountryName_South_Korea', 'RegionCountryName_Spain',
                                'RegionCountryName_Sweden', 'RegionCountryName_Switzerland',
                                'RegionCountryName_Syria', 'RegionCountryName_Taiwan',
                                'RegionCountryName_Thailand', 'RegionCountryName_Turkmenistan',
                                'RegionCountryName_United_Kingdom', 'RegionCountryName_United_States',
                                'RegionCountryName_the_Netherlands',]
            ),
            FixedCombinations(
                column_names=['StoreKey', 'StoreType_Catalog', 'StoreType_Online',
                                'StoreType_Reseller', 'StoreType_Store', 'SellingAreaSize']
            ),
            FixedCombinations(
                column_names=['DiscountAmount', 'DiscountPercent', 'DiscountQuantity']
            ),
            OneHotEncoding(
                column_names=['BrandName_A_Datum', 'BrandName_Adventure_Works', 'BrandName_Contoso',
                                'BrandName_Fabrikam', 'BrandName_Litware', 'BrandName_NA',
                                'BrandName_Northwind_Traders', 'BrandName_Proseware',
                                'BrandName_Southridge_Video', 'BrandName_Tailspin_Toys',
                                'BrandName_The_Phone_Company', 'BrandName_Wide_World_Importers',
                                'RegionCountryName_Armenia',
                                'RegionCountryName_Australia', 'RegionCountryName_Bhutan',
                                'RegionCountryName_Canada', 'RegionCountryName_China',
                                'RegionCountryName_Denmark', 'RegionCountryName_France',
                                'RegionCountryName_Germany', 'RegionCountryName_Greece',
                                'RegionCountryName_India', 'RegionCountryName_Iran',
                                'RegionCountryName_Ireland', 'RegionCountryName_Italy',
                                'RegionCountryName_Japan', 'RegionCountryName_Kyrgyzstan',
                                'RegionCountryName_Malta', 'RegionCountryName_Pakistan',
                                'RegionCountryName_Poland', 'RegionCountryName_Portugal',
                                'RegionCountryName_Romania', 'RegionCountryName_Russia',
                                'RegionCountryName_Singapore', 'RegionCountryName_Slovenia',
                                'RegionCountryName_South_Korea', 'RegionCountryName_Spain',
                                'RegionCountryName_Sweden', 'RegionCountryName_Switzerland',
                                'RegionCountryName_Syria', 'RegionCountryName_Taiwan',
                                'RegionCountryName_Thailand', 'RegionCountryName_Turkmenistan',
                                'RegionCountryName_United_Kingdom', 'RegionCountryName_United_States',
                                'RegionCountryName_the_Netherlands', 'StoreType_Catalog', 'StoreType_Online',
                                'StoreType_Reseller', 'StoreType_Store',
                                'IsWorkDay_WeekEnd', 'IsWorkDay_WorkDay'
                ]
            )
        ]

        self.model = GaussianCopulaSynthesizer(
            self.metadata,
            enforce_min_max_values=True,
            enforce_rounding=False,
            numerical_distributions={
                #####
            },
            default_distribution='norm'
        )

        self.model.add_constraints(constraints=[self.constraints])

    def create_facts(min_date, max_date):
        """
        Creates known facts about the provided date range. Facts are:
        DateKey, CalendarYear, CalendarQuarterLabel, CalendarWeekLabel,
        IsWorkDay, MonthNumber, CalendarDayOfWeekNumber
        """
        daterange = pd.date_range(min_date, max_date - pd.Timedelta(days=1), freq='d').strftime('%Y-%m-%d').tolist()


    def fit(self, data):
        self.model = self.model.fit(data)
        self.model.save(
            filepath=path.join(local_synthmodel_path, 'CPSynthesizer.pkl')
        )

    def sample(self):
        """
        Create synthetic data for the the defined date range, based on existing data
        """
        self.model.sample()

    def sample_with_facts(self, factsdata):
        """
        Create synthetic data for the the defined date range, based on existing data
        and using known facts
        """
        self.model.sample_remaining_columns(known_columns=factsdata)

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
            self.__init__()

    def evaluate(self, actual_data, synth_data):
        from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
        from sdv.evaluation.single_table import get_column_plot

        # 1. perform basic validity checks
        diagnostic = run_diagnostic(actual_data, synth_data, self.metadata)

        # 2. measure the statistical similarity
        quality_report = evaluate_quality(actual_data, synth_data, self.metadata)

        # 3. plot the data
        fig = get_column_plot(
            real_data=actual_data,
            synthetic_data=synth_data,
            metadata=self.metadata,
            column_name='SalesAmount'
        )

        fig.show()
