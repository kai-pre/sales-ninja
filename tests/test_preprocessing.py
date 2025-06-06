import pandas as pd
from salesninja.data import SalesNinja
from salesninja.ml.preprocessor import preprocess_features

df = SalesNinja().get_ml_data()
df_new = preprocess_features(df, simple = True)
print(df_new)
