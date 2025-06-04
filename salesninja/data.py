### Imports
import pandas as pd



### Class definition

class SalesNinja():
    def get_data(self):
        data = pd.read_csv("../raw_data/FactSales.csv")
        data.set_index(['SalesKey'], inplace=True)

        ### TO DO:
        ### join appropriate extra tables:
        ### DimStore
        ### DimGeography (only reachable by DimStore)
        ### DimProduct
        ### DimProductSubcategory (only reachable by DimProduct)
        ### DimProductCategory (only reachable by DimProductSubcategory)
        ### DimPromotion
        #data = pd.merge(data, pd.read_csv("../raw_data/DimStore.csv"), on = "StoreKey")

        return data
