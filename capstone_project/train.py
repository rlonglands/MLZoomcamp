
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('insurance.csv')

df_full_train, df_test = train_test_split(df, test_size=0.2)
y_full_train = np.log1p(df_full_train.charges.values)
dv = DictVectorizer(sparse=False)
del df_full_train['charges']
full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)

dt = DecisionTreeRegressor(min_samples_leaf=35, random_state=1, max_depth = 6)
dt.fit(X_full_train, y_full_train)
model = dt.fit(X_full_train, y_full_train)

with open("model.pkl", "wb") as f:
    pickle.dump((model), f)

with open("dv.bin", "wb") as f:
    pickle.dump((dv), f)   



