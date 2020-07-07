import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle

import lightgbm as lgb

df = pd.read_csv('京兴20191209.csv')

label_columns = list(filter(lambda x: 'F_S' in x, df.columns))
input_columns = list(filter(lambda x: 'F_S' not in x and 'DateTime' not in x, df.columns))
# df = df[df['F_104303.F_S'] != 0]
# df = df[df['F_104304.F_S'] != 0]
X = df.loc[:, input_columns]
X = X.iloc[:, 1:].values
y = df.loc[:, label_columns]
y = (y.iloc[:, 0] + y.iloc[:, 1]).values

# for i in range(len(X.columns)):
#     X.iloc[:, i: i+1] = Normalizer().fit_transform(X.iloc[:, i: i+1])

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6000, random_state=42)

x_normalizer = StandardScaler()
normed_X_train = x_normalizer.fit_transform(X_train)
normed_X_test = x_normalizer.transform(X_test)
y_normalizer = StandardScaler()
normed_y_train = y_normalizer.fit_transform(y_train.reshape(-1, 1))
normed_y_test = y_normalizer.transform(y_test.reshape(-1, 1))
layers_list = [2, 3, 4, 5, 6]
activation_func_list = [F.relu, F.tanh, F.sigmoid, F.leaky_relu, F.elu]
hidden_list = [16, 32, 64, 128, 256]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64, layers=3, activation_func=F.leaky_relu, use_bn=True):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.op_list = []
        self.layers = layers
        for i in range(layers):
            if i == 0:
                self.linear = nn.Linear(in_features=input_dim, out_features=hidden)
            elif i == layers - 1:
                self.linear = nn.Linear(in_features=hidden, out_features=output_dim)
            else:
                self.linear = nn.Linear(in_features=hidden, out_features=hidden)
            self.modules_list.append(self.linear)
            self.op_list.append(self.linear)

            if use_bn and i != layers - 1:
                self.bn = nn.BatchNorm1d(num_features=hidden)
                self.modules_list.append(self.bn)
                self.op_list.append(self.bn)
            self.op_list.append(activation_func)

    def forward(self, x:torch.Tensor):
        for op in self.op_list:
            x = op(x)
        return x


def eval(normed_pred, normed_y):
    R_2 = 1.0 - mean_squared_error(normed_pred, normed_y)
    pred = y_normalizer.inverse_transform(normed_pred).reshape(-1)
    y = y_normalizer.inverse_transform(normed_y).reshape(-1)
    MSE = mean_squared_error(pred, y)
    MRE = np.mean(np.abs(pred[y != 0] - y[y != 0])/np.abs(y[y != 0]))
    return {'MSE':MSE, 'R^2':R_2, 'MRE':MRE}

y_mean = np.mean(normed_y_train)
print('baseline training mse:', eval(normed_pred=np.ones(normed_y_train.shape, dtype=np.float32)*y_mean, normed_y=normed_y_train))
print('baseline testing mse:', eval(normed_pred=np.ones(normed_y_test.shape, dtype=np.float32)*y_mean, normed_y=normed_y_test))

print('linear model')
model = LinearRegression().fit(normed_X_train, normed_y_train)
out = model.predict(normed_X_train)
print('training: ', eval(out, normed_y_train))
out = model.predict(normed_X_test)
print('testing: ', eval(out, normed_y_test))
with open('normalizer.pickle', 'wb') as f:
    pickle.dump((x_normalizer, y_normalizer), f)
with open('linear_model.pickle', 'wb') as f:
    pickle.dump(model, f)



def eval(normed_pred, normed_y):
    R_2 = 1.0 - mean_squared_error(normed_pred, normed_y)
    pred = y_normalizer.inverse_transform(normed_pred).reshape(-1)
    y = y_normalizer.inverse_transform(normed_y).reshape(-1)
    MSE = mean_squared_error(pred, y)
    MRE = np.mean(np.abs(pred[y != 0] - y[y != 0])/np.abs(y[y != 0]))
    return {'MSE':MSE, 'R^2':R_2, 'MRE':MRE}

def mini_batch(batch_size=6000):
    start_idx = 0
    N = normed_X_train.shape[0]
    while start_idx < N:
        yield normed_X_train[start_idx:min(start_idx + batch_size, normed_X_train.shape[0])], normed_y_train[start_idx:min(start_idx + batch_size, normed_X_train.shape[0])]
        start_idx += batch_size
store = {}
def train(epochs=500, validation_steps=300, USE_CUDA=True, cuda_id=0):
    device = torch.device('cuda:%d'%cuda_id) if USE_CUDA else torch.device('cpu')

    model_list = [('layers', 2, MLP(layers=2,input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('layers', 3, MLP(layers=3,input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('layers', 4, MLP(layers=4,input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('layers', 5, MLP(layers=5,input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('layers', 6, MLP(layers=6,input_dim=normed_X_train.shape[1], output_dim=1)),

                  ('activation_func', 'F.relu', MLP(activation_func=F.relu,input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('activation_func', 'F.tanh', MLP(activation_func=F.tanh, input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('activation_func', 'F.sigmoid', MLP(activation_func=F.sigmoid, input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('activation_func', 'F.leaky_relu', MLP(activation_func=F.leaky_relu, input_dim=normed_X_train.shape[1], output_dim=1)),
                  ('activation_func', 'F.elu', MLP(activation_func=F.elu, input_dim=normed_X_train.shape[1], output_dim=1)),

                  ('hidden', 16, MLP(input_dim=normed_X_train.shape[1], output_dim=1, hidden=16)),
                  ('hidden', 32, MLP(input_dim=normed_X_train.shape[1], output_dim=1, hidden=32)),
                  ('hidden', 64, MLP(input_dim=normed_X_train.shape[1], output_dim=1, hidden=64)),
                  ('hidden', 128, MLP(input_dim=normed_X_train.shape[1], output_dim=1, hidden=128)),
                  ('hidden', 256, MLP(input_dim=normed_X_train.shape[1], output_dim=1, hidden=256))]

    # model = MLP(input_dim=normed_X_train.shape[1], output_dim=1)
    for parameter_name, parameter_type, model in model_list:
        criterion = nn.MSELoss()
        if USE_CUDA:
            model = model.to(device)
            criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5000)
        # writer = SummaryWriter('log')
        step = 0

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in mini_batch():
                x_tensor = torch.from_numpy(x_batch).float().to(device)
                y_tensor = torch.from_numpy(y_batch).float().to(device)
                x_tensor.requires_grad_(True)
                pred = model(x_tensor)
                loss = criterion(input=pred, target=y_tensor)
                # print(
                #     f'training epoch = {epoch}, loss = {loss.detach().cpu().item()}, eval = {eval(normed_pred=pred.cpu().detach().numpy(), normed_y=y_batch)}')
                # writer.add_scalar(tag='loss/train', scalar_value=loss.detach().cpu().item(), global_step=step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(step)

                if step % validation_steps == 0:
                    model.eval()
                    x_tensor = torch.from_numpy(normed_X_test).float().to(device)
                    y_tensor = torch.from_numpy(normed_y_test).float().to(device)
                    pred = model(x_tensor)
                    loss = criterion(input=pred, target=y_tensor)
                    print(
                        f'validation epoch = {epoch}, loss = {loss.detach().cpu().item()}, eval = {eval(normed_pred=pred.cpu().detach().numpy(), normed_y=normed_y_test)}')
                    # writer.add_scalar(tag='loss/validation', scalar_value=loss.detach().cpu().item(), global_step=step)
                    # writer.add_scalar(tag='lr', scalar_value=lr_scheduler.get_lr()[0], global_step=step)
                step += 1
        model.eval()
        x_tensor = torch.from_numpy(normed_X_test).float().to(device)
        pred = model(x_tensor)
        out = pred.detach().cpu().numpy()
        eval_res = eval(out, normed_y_test)
        if parameter_name not in store:
            store[parameter_name] = []
        store[parameter_name].append((parameter_name, parameter_type, eval_res))
        print('testing: ', eval_res)

        # torch.save(model.state_dict(), 'NeuralNetwork.torch_model')
    with open('NN_result.pickle', 'wb') as f:
        pickle.dump(store, f)
train()
