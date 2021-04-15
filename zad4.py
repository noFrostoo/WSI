import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, plot_roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv("adult.data.csv")
data_test = pd.read_csv("adult.test.csv")

data = data.replace(" ?", np.NaN)
data = data.dropna()

data_test = data_test.replace(" ?", np.NaN)
data_test = data_test.dropna()


lbe = LabelEncoder()

nonnumerical_collumns = data.select_dtypes(include=['object'])
nonnumerical_collumns = nonnumerical_collumns.apply(lbe.fit_transform)

nonnumerical_collumns_test = data_test.select_dtypes(include=['object'])
nonnumerical_collumns_test = nonnumerical_collumns_test.apply(lbe.fit_transform)

data = data.drop(nonnumerical_collumns.columns, axis=1)
data = pd.concat([data, nonnumerical_collumns], axis=1)

data_test = data_test.drop(nonnumerical_collumns_test.columns, axis=1)
data_test = pd.concat([data_test, nonnumerical_collumns_test], axis=1)

x_train = data.drop(" income_group", axis=1)
y_train = data[" income_group"]

x_test = data_test.drop(" income_group", axis=1)
y_test = data_test[" income_group"]

# rfc = RandomForestClassifier(n_estimators=500, max_features="sqrt")
rfc = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=3, max_depth=14)
# rfc = RandomForestClassifier(n_estimators=500, max_features=3, max_depth=14)
# rfc = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=3)

rfc.fit(x_train, y_train)

predict_y = rfc.predict(x_test)
print(accuracy_score(y_test, predict_y))

plot_roc_curve(rfc, x_test, y_test)
plt.show()


rcf2 = RandomForestClassifier()

# kfold = KFold(n_splits=3)
# numberOfEstimators = [350]
# numberOfFeatures = range(1, 8, 1)
# MiniumSambleLeaf = range(1, 8, 1)
# Depth = range(1, 15, 1)
# gridDirt = {
#     'n_estimators': numberOfEstimators,
#     'max_features': numberOfFeatures,
#     'min_samples_leaf': MiniumSambleLeaf,
#     'max_depth': Depth
# }
# grid = GridSearchCV(estimator=rcf2, param_grid=gridDirt, cv=kfold)
# result = grid.fit(x_train, y_train)
# print(result.best_params_)
# print(result.best_score_)