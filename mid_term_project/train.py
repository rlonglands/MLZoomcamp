
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

df = pd.read_csv('mid_term_project\cc_approvals.csv')

# add column names to dataframe
df.columns = ['gender', 'age', 'debt', 'married', 'bankcustomer', 'educationlevel',
       'ethnicity', 'yearsemployed', 'priordefault', 'employed', 'creditscore',
       'driverslicense', 'citizen', 'zipcode', 'income', 'approvalstatus']

# replace problem characters with integers
df.approvalstatus = df.approvalstatus.replace('+', 1)
df.approvalstatus = df.approvalstatus.replace('-', 0)
df = df.replace('?',0 )

df["age"] = pd.to_numeric(df["age"])
df["zipcode"] = pd.to_numeric(df["zipcode"])

# Split the dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=2)

target = 'approvalstatus'
y_full_train = df_full_train[target]
y_test = df_test[target]

del df_test[target]
del df_full_train[target]

# model is more accurate with ethicity removed
delete = 'ethnicity'
del df_test[delete]
del df_full_train[delete]

dv = DictVectorizer()
full_train_dicts = df_full_train.fillna(0).to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_test, label=y_test)
xgb_params = {
        'eta': 0.1, 
        'max_depth': 1,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)

with open("mid_term_project\cc_model.pkl", "wb") as f:
    pickle.dump(model, f)



