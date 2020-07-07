import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error
import pickle

import lightgbm as lgb

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
X = df.loc[:, input_columns]
X = X.iloc[:, 1:].values
y = df.loc[:, label_columns]
y = (y.iloc[:, 0] + y.iloc[:, 1]).values

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6000, random_state=42)

x_normalizer = StandardScaler()
normed_X_train = x_normalizer.fit_transform(X_train)
normed_X_test = x_normalizer.transform(X_test)
y_normalizer = StandardScaler()
normed_y_train = y_normalizer.fit_transform(y_train.reshape(-1, 1))
normed_y_test = y_normalizer.transform(y_test.reshape(-1, 1))

print('gbdt model')
training_data = lgb.Dataset(data=normed_X_train, label=normed_y_train.squeeze())
testing_data = lgb.Dataset(data=normed_X_test, label=normed_y_test.squeeze())
store = {}

params = {'objective':'regression', 'num_iterations':150, 'learning_rate':0.3, 'num_leaves':100000, 'metric':'l2'}
store['num_leaves'] = []
for num_leaves in [10, 100, 1000, 10000, 100000]:
    params['num_leaves'] = num_leaves
    rt = lgb.train(params=params, train_set=training_data, valid_sets=[training_data, testing_data], early_stopping_rounds=100)
    model = rt
    out = model.predict(normed_X_test)
    eval_res = eval(out, normed_y_test)
    store['num_leaves'].append(eval_res)
    print('testing: ', eval_res)
params['num_leaves'] = 100000

store['learning_rate'] = []
for learning_rate in [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]:
    params['learning_rate'] = learning_rate
    rt = lgb.train(params=params, train_set=training_data, valid_sets=[training_data, testing_data], early_stopping_rounds=100)
    model = rt
    out = model.predict(normed_X_test)
    eval_res = eval(out, normed_y_test)
    store['learning_rate'].append(eval_res)
    print('testing: ', eval_res)
params['num_leaves'] = 100000

with open('gbdt_result.pickle', 'wb') as f:
    pickle.dump(store, f)
# rt = lgb.train(params=params, train_set=training_data, valid_sets=[training_data, testing_data],)
# model = rt
# out = model.predict(normed_X_train)
# print('training: ', eval(out, normed_y_train))
# out = model.predict(normed_X_test)
# print('testing: ', eval(out, normed_y_test))
# exit()