from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from xgboost import XGBRegressor

train_x = np.array(pd.read_csv('train_x.csv'))
train_y = np.array(pd.read_csv('train_y.csv')).ravel()

scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

param_grid = dict()
param_grid['lambda'] = [0.5]
param_grid['alpha'] = [0.1]
param_grid['subsample'] = np.arange(0.1, 1, 0.1)
param_grid['eta'] = np.arange(0, 0.4, 0.05)
param_grid['sampling_method'] = ['gradient_based']

grid_search = GridSearchCV( estimator=XGBRegressor(max_depth=3, tree_method='gpu_hist'), param_grid=param_grid, n_jobs=-1, cv=12)
grid_search.fit(train_x, train_y)
best_est = grid_search.best_estimator_
best_score = grid_search.best_score_
best_params = grid_search.best_params_

print(best_params)
print(best_score)

test_x = pd.read_csv('test_x.csv')
ids = test_x['Id']
test_x = np.array(test_x.drop(['Id'], axis='columns'))

test_x = scaler.transform(test_x)
predictions_ = best_est.predict(test_x).reshape(-1)

results = pd.DataFrame()
results['Id'] = ids
results['SalePrice'] = predictions_
results.to_csv('predictions.csv', header=True, index=False)