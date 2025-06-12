import pandas as pd
from salesninja.data import SalesNinja
from salesninja.ml.preprocessing import preprocess_features, seasonalize_data

df = SalesNinja().get_ml_data()
df_new = preprocess_features(df, simple = True)
df_new = seasonalize_data(df_new)
print(df_new)
