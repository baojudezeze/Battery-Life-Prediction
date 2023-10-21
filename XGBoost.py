# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import numpy as np

# load data
dataset = loadtxt('./XGBoost/2020minmax_WP.txt', delimiter=",", encoding='utf-16')
# split data into X and y
for j in range(len(dataset)):
    if dataset[j, 9] > 0.01:
        dataset[j, 9] = 1
    else:
        dataset[j, 9] = 0

X = dataset[:, 0:9]
y = dataset[:, 9]

# y = np.arange(0, 8621, 1)
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
