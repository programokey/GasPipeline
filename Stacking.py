from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np
import itertools
from sklearn.metrics.regression import mean_squared_error
import pandas as pd

class LGBStacking:
    def __init__(self, lgb_params, stacking_params=None):
        self.stacking_params = stacking_params if stacking_params is not None else {}
        self.lgb_params = lgb_params

    @staticmethod
    def get_intermediate_representation(rt, X):
        # type:(lgb.Booster, np.ndarray)->np.ndarray
        intermediate_representation = np.zeros((X.shape[0], rt.num_trees()), dtype=np.float32)
        pred_leaves = rt.predict(X, pred_leaf=True)
        for i, j in itertools.product(range(X.shape[0]), range(rt.num_trees())):
            intermediate_representation[i, j] = rt.get_leaf_output(tree_id=j, leaf_id=pred_leaves[i, j])
        return intermediate_representation

    def eval(self, rt, X, y, metric='L2'):
        # type: (lgb.Booster, np.ndarray, np.ndarray, str)->np.ndarray
        y_pred = rt.predict(data=X)
        if metric == 'L2':
            return np.mean(np.square(y_pred - y))

    def train(self, X, y, validate_size=0.25, random_state=None, shuffle=True, categorical_feature='auto'):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=validate_size, random_state=random_state,
                                                            shuffle=shuffle)
        self.layers = []
        performance = []
        min_loss = 1e9
        count = 0
        best_round = 0
        for layers in range(self.stacking_params.get('maximum_layers', 5)):
            if len(self.layers) > 0:
                rt = self.layers[-1]
                train_X_intermediate = LGBStacking.get_intermediate_representation(rt, train_X)
                test_X_intermediate = LGBStacking.get_intermediate_representation(rt, test_X)
                train_X = np.hstack((train_X, train_X_intermediate))
                test_X = np.hstack((test_X, test_X_intermediate))
            training_data = lgb.Dataset(data=train_X, label=train_y.squeeze(), categorical_feature=categorical_feature)
            testing_data = lgb.Dataset(data=test_X, label=test_y.squeeze(), categorical_feature=categorical_feature)
            rt = lgb.train(params=self.lgb_params, train_set=training_data, valid_sets=[training_data, testing_data])
            self.layers.append(rt)
            performance.append(self.eval(rt, test_X, test_y))
            if min_loss > performance[-1]:
                count = 0
                min_loss = performance[-1]
                best_round = layers
            else:
                count += 1
                if count > self.stacking_params.get('min_early_stopping_layers', 0) \
                        and count == self.stacking_params.get('early_stopping_round', 0):
                    self.layers = self.layers[:best_round + 1]
                    return performance

    def predict(self, X):
        for i, rt in enumerate(self.layers):
            if i < len(self.layers) - 1:
                X_intermediate = LGBStacking.get_intermediate_representation(rt, X)
                X = np.hstack((X, X_intermediate))
            else:
                y = rt.predict(X)
                return y

def eval(normed_pred, normed_y):
    print(normed_pred.shape, normed_y.shape)
    R_2 = 1.0 - mean_squared_error(normed_pred, normed_y)
    pred = y_normalizer.inverse_transform(normed_pred).reshape(-1)
    y = y_normalizer.inverse_transform(normed_y).reshape(-1)
    MSE = mean_squared_error(pred, y)
    MRE = np.mean(np.abs(pred[y != 0] - y[y != 0])/np.abs(y[y != 0]))
    return {'MSE':MSE, '1 - R^2':1 - R_2, 'MRE':MRE}

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


learning_rates  = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]

result = [{'MSE': 1210.1215466089106, '1 - R^2': 6.4230182728053364e-06, 'MRE': 0.001027962797433858},
{'MSE': 1129.783971186976, '1 - R^2': 5.996606796787596e-06, 'MRE': 0.0005665661292965678},
{'MSE': 1336.2736553790626, '1 - R^2': 7.09260167297554e-06, 'MRE': 0.0007171878749659522},
{'MSE': 1088.406993922103, '1 - R^2': 5.7769882949587625e-06, 'MRE': 0.0006001504962409913},
{'MSE': 880.5374090118733, '1 - R^2': 4.6736692557169945e-06, 'MRE': 0.0005386696360755044},
{'MSE': 1197.04569096991, '1 - R^2': 6.3536149472742665e-06, 'MRE': 0.0007382772370375937}]
import matplotlib.pyplot as plt
def draw_line_chart(x, y, x_label, y_label, file, format='%.g'):
    plt.scatter(x=x, y=y, c='r')
    # plt.plot(x, y, scalex='log')
    plt.ylim(4e-6, 8e-6)
    plt.plot(x, y)
    plt.yscale('log')
    for x, y in zip(x, y):
        plt.text(x, y, format%y, ha = 'center',va = 'bottom',fontsize=13)
    # plt.xscale('log')
    plt.xlabel(x_label, fontsize=17)
    plt.ylabel(y_label, fontsize=17)
    plt.savefig(file)
    plt.close()
    # plt.show()
# mse = [item['MSE'] for item in result]
# draw_line_chart(learning_rates, mse, x_label='learning rate', y_label='MSE', file='fig/stacking_MSE.png', format='%.2g')
# mre = [item['MRE'] for item in result]
# draw_line_chart(learning_rates, mre, x_label='learning rate', y_label='MRE', file='fig/stacking_MRE.png', format='%.g')
R_2 = [item['1 - R^2'] for item in result]
draw_line_chart(learning_rates, R_2, x_label='learning rate', y_label='1 - r^2', file='fig/stacking_R_2.png')

exit()
params = {'objective':'regression', 'num_iterations':128, 'learning_rate':0.10, 'num_leaves':1000, 'metric':'l2', 'early_stopping_round':10}
# stacking = LGBStacking(lgb_params=params)
# stacking.train(X=normed_X_train, y=normed_y_train)
# print(eval(stacking.predict(normed_X_test), normed_y_test))
res = []
for learning_rate in [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]:
    params['learning_rate'] = learning_rate
    stacking = LGBStacking(lgb_params=params)
    performance = stacking.train(X=normed_X_train, y=normed_y_train)
    res.append((performance, eval(stacking.predict(normed_X_test), normed_y_test)))
    # print(eval(stacking.predict(normed_X_test), normed_y_test))
    # print(performance)

for i, learning_rate in enumerate([0.05, 0.1, 0.15, 0.20, 0.25, 0.30]):
    print('learning_rate = %f--------------'%learning_rate)
    print(res[i][0])
    print(res[i][1])
