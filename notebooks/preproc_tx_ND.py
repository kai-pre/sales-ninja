import pandas as pd
import os

print(os.getcwd())


def get_ml_data():
    # Master table
    df = pd.read_csv("raw_data/FactSales.csv").drop(['CurrencyKey'], axis=1)
    df = df.merge(
        pd.read_csv(
            "raw_data/DimPromotion.csv",
            usecols=["DiscountPercent", "PromotionKey"],
        ),
        on="PromotionKey",
        how="left",
    )
    
    
 
    df = df.merge(
        pd.read_csv(
            "raw_data/DimDate.csv",
            usecols=[
                "DateKey",
                "IsWorkDay",
                "CalendarWeekLabel",
                "CalendarYear",
                "MonthNumber",
                "CalendarQuarterLabel",
                "CalendarDayOfWeekNumber",
            ],
        ),
        on="DateKey",
        how="left",
    )

    
    # product hierarchy merge
    prodsubcat = pd.read_csv(
        "raw_data/DimProductSubcategory.csv",
        usecols=["ProductSubcategoryKey", "ProductCategoryKey"],
    ).merge(
        pd.read_csv(
            "raw_data/DimProductCategory.csv",
            usecols=["ProductCategoryKey"],
        ),
        on="ProductCategoryKey",
        how="left",
    )
    prod = pd.read_csv(
        "raw_data/DimProduct.csv",
        usecols=[
            "ProductKey",
            "ProductSubcategoryKey",
            "BrandName",
            "ClassID",
            "StyleID",
            "ColorID",
            "Weight",
            "WeightUnitMeasureID",
            "StockTypeID",
        ],
    ).merge(prodsubcat, on="ProductSubcategoryKey", how="left")
    df = df.merge(prod, on="ProductKey", how="left")
    

    # store hierarchy merge
    store = pd.read_csv(
        "raw_data/DimStore.csv",
        usecols=[
            "StoreKey",
            "GeographyKey",
            "StoreType",
            "EmployeeCount",
            "SellingAreaSize",
        ],
    ).merge(
        pd.read_csv(
            "raw_data/DimGeography.csv",
            usecols=[
                "GeographyKey",
                "GeographyType",
                "ContinentName",
                "CityName",
                "StateProvinceName",
                "RegionCountryName",
            ],
        ),
        on="GeographyKey",
        how="left",
    )
    df = df.merge(store, on="StoreKey", how="left")
    return df


def get_dashboard_data():
    # Master table
    df = pd.read_csv("raw_data/FactSales.csv").drop(['CurrencyKey'], axis=1)
    

    df = df.merge(
    pd.read_csv(
        "raw_data/DimChannel.csv", usecols=["ChannelKey", "ChannelName"]
    ),
    left_on="channelKey",
    right_on="ChannelKey",
    how="left",
   )
    

# Drop 'channelKey' column if it exists
    if 'channelKey' in df.columns:
        df.drop('channelKey', axis=1, inplace=True)

# Rename 'ChannelKey.1' to 'ChannelKey' if it exists
    if 'ChannelKey.1' in df.columns:
        df.rename(columns={'ChannelKey.1': 'ChannelKey'}, inplace=True)

    
    df = df.merge(
        pd.read_csv(
            "raw_data/DimPromotion.csv",
            usecols=["PromotionKey", "PromotionName", "PromotionType"],
        ),
        on="PromotionKey",
        how="left",
    )
    

    df = df.merge(
        pd.read_csv(
            "raw_data/DimDate.csv",
            usecols=[
                "DateKey",
                "CalendarYear",
                "CalendarMonthLabel",
                "MonthNumber",
                "CalendarQuarterLabel",
                "CalendarDayOfWeekNumber",
                "CalendarDayOfWeekLabel",
            ],
        ),
        on="DateKey",
        how="left",
    )
    
      # product hierarchy merge
    prodsubcat = pd.read_csv(
        "raw_data/DimProductSubcategory.csv",
        usecols=[
            "ProductSubcategoryKey",
            "ProductSubcategoryName",
            "ProductCategoryKey",
        ],
    ).merge(
        pd.read_csv(
            "raw_data/DimProductCategory.csv",
            usecols=["ProductCategoryKey", "ProductCategoryName"],
        ),
        on="ProductCategoryKey",
        how="left",
    )
    prod = pd.read_csv(
        "raw_data/DimProduct.csv",
        usecols=["ProductKey", "ProductName", "ProductSubcategoryKey"],
    ).merge(prodsubcat, on="ProductSubcategoryKey", how="left")
    df = df.merge(prod, on="ProductKey", how="left")

    # store hierarchy merge
    store = pd.read_csv(
        "raw_data/DimStore.csv",
        usecols=["StoreKey", "GeographyKey", "StoreType", "StoreName"],
    ).merge(
        pd.read_csv(
            "raw_data/DimGeography.csv",
            usecols=[
                "GeographyKey",
                "GeographyType",
                "ContinentName",
                "CityName",
                "StateProvinceName",
                "RegionCountryName"
            ],
        ),
        on="GeographyKey",
        how="left",
    )
    df = df.merge(store, on="StoreKey", how="left")
    return df

if __name__ == "__main__":

    # show all columns in preview(and not just 1st & last columns)
    pd.set_option("display.max_columns", None)

    #created final tables and view top 10 rows
    #merged_fc = get_ml_data()
    #print("Preview of merged data for forecast:")
    #print(merged_fc.head(10))

    merged_act = get_dashboard_data()
    print("Preview of merged data for forecast:")
    print(merged_act.head(10))

    # generate a csv to process it further in jupyter notebook
    #merged_fc.to_csv("data_ml_merged.csv", index=False)
    #print("Data saved to 'data_ml_merged.csv'")

    merged_act.to_csv("data_dashboard_merged.csv", index=False)
    print("Data saved to 'data_dashboard_merged.csv'")

    
