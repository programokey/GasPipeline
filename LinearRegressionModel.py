import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors.regression import KNeighborsRegressor

import pickle

def eval(normed_pred, normed_y):
    R_2 = 1.0 - mean_squared_error(normed_pred, normed_y)
    pred = y_normalizer.inverse_transform(normed_pred).reshape(-1)
    y = y_normalizer.inverse_transform(normed_y).reshape(-1)
    MSE = mean_squared_error(pred, y)
    MRE = np.mean(np.abs(pred[y != 0] - y[y != 0])/np.abs(y[y != 0]))
    return {'MSE':MSE, 'R^2':R_2, 'MRE':MRE}

df = pd.read_csv('京兴20191209.csv')
# df = pd.read_csv('国华20200319.csv')

label_columns = list(filter(lambda x: 'F_S' in x, df.columns))
input_columns = list(filter(lambda x: 'F_S' not in x and 'DateTime' not in x, df.columns))
print(label_columns)
X = df.loc[:, input_columns]
X = X.iloc[:, 1:].values
y = df.loc[:, label_columns]
y = (y.iloc[:, 0] + y.iloc[:, 1]).values

# import matplotlib.pyplot as plt
# plt.plot(y)
# plt.show()
# exit()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6000, random_state=42)

x_normalizer = StandardScaler()
normed_X_train = x_normalizer.fit_transform(X_train)
normed_X_test = x_normalizer.transform(X_test)
y_normalizer = StandardScaler()
normed_y_train = y_normalizer.fit_transform(y_train.reshape(-1, 1))
normed_y_test = y_normalizer.transform(y_test.reshape(-1, 1))

print('gbdt model')
model_list = [('LinearRegression', None, None, LinearRegression()),

              ('Lasso', 'alpha', 1e-6, Lasso(alpha=1e-5)),
              ('Lasso', 'alpha', 1e-5, Lasso(alpha=1e-5)),
              ('Lasso', 'alpha', 1e-4, Lasso(alpha=1e-4)),
              ('Lasso', 'alpha', 1e-3, Lasso(alpha=1e-3)),
              ('Lasso', 'alpha', 1e-2, Lasso(alpha=1e-2)),
              ('Lasso', 'alpha', 1e-1, Lasso(alpha=1e-1)),
              ('Lasso', 'alpha', 1.0, Lasso(alpha=1.0)),

              ('Ridge', 'alpha', 1e-6, Ridge(alpha=1e-5)),
              ('Ridge', 'alpha', 1e-5, Ridge(alpha=1e-5)),
              ('Ridge', 'alpha', 1e-4, Ridge(alpha=1e-4)),
              ('Ridge', 'alpha', 1e-3, Ridge(alpha=1e-3)),
              ('Ridge', 'alpha', 1e-2, Ridge(alpha=1e-2)),
              ('Ridge', 'alpha', 1e-1, Ridge(alpha=1e-1)),
              ('Ridge', 'alpha', 1.0,  Ridge(alpha=1.0)),

              ('KNeighborsRegressor', 'n_neighbors', 7, KNeighborsRegressor(n_neighbors=7)),
              ('KNeighborsRegressor', 'n_neighbors', 6, KNeighborsRegressor(n_neighbors=6)),
              ('KNeighborsRegressor', 'n_neighbors', 5, KNeighborsRegressor(n_neighbors=5)),
              ('KNeighborsRegressor', 'n_neighbors', 4, KNeighborsRegressor(n_neighbors=4)),
              ('KNeighborsRegressor', 'n_neighbors', 3, KNeighborsRegressor(n_neighbors=3)),
              ('KNeighborsRegressor', 'n_neighbors', 2, KNeighborsRegressor(n_neighbors=2)),
              ('KNeighborsRegressor', 'n_neighbors', 1, KNeighborsRegressor(n_neighbors=1)),


              ('KNeighborsRegressor_distance', 'n_neighbors', 7, KNeighborsRegressor(n_neighbors=7, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 6, KNeighborsRegressor(n_neighbors=6, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 5, KNeighborsRegressor(n_neighbors=5, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 4, KNeighborsRegressor(n_neighbors=4, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 3, KNeighborsRegressor(n_neighbors=3, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 2, KNeighborsRegressor(n_neighbors=2, weights='distance')),
              ('KNeighborsRegressor_distance', 'n_neighbors', 1, KNeighborsRegressor(n_neighbors=1, weights='distance')),


              ('SVR', 'kernel', 'rbf', SVR(kernel='rbf')),
              ('SVR', 'kernel', 'linear', SVR(kernel='linear')),
              ('SVR', 'kernel', 'poly', SVR(kernel='poly')),]
store = {}
for model_type, parameter_name, parameter_type, model in model_list:
    model.fit(normed_X_train, normed_y_train)
    out = model.predict(normed_X_test)
    eval_res = eval(out, normed_y_test)
    if model_type not  in store:
        store[model_type] = []
    store[model_type].append((parameter_name, parameter_type, eval_res))
    print('testing: ', eval_res)

with open('regression_result.pickle', 'wb') as f:
    pickle.dump(store, f)
