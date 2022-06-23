import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


df_train["TrainFlag"] = True
df_test["TrainFlag"] = False

df_all = df_train.append(df_test)
df_all.index = df_all["Id"]
df_all.drop("Id", axis = 1, inplace = True)

df_all = pd.get_dummies(df_all, drop_first=True)

df_train = df_all[df_all["TrainFlag"] == True]
df_train = df_train.drop(["TrainFlag"], axis = 1)

df_test = df_all[df_all["TrainFlag"] == False]
df_test = df_test.drop(["TrainFlag"], axis = 1)
df_test = df_test.drop(["SalePrice"], axis = 1)


y = df_train["SalePrice"].values
X = df_train.drop("SalePrice", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dtest = xgb.DMatrix(df_test.values)

params = {'objective': 'reg:squarederror','silent':1, 'random_state':1234, 'eval_metric': 'rmse',}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(params, dtrain, num_round, early_stopping_rounds=20, evals=watchlist)


prediction_XG = model.predict(dtest, ntree_limit = model.best_ntree_limit)
prediction_XG = np.round(prediction_XG)

submission = pd.DataFrame({"id": df_test.index, "SalePrice": prediction_XG})
