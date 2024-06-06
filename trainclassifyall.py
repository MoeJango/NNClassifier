import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from scipy.stats import zscore
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features=1041, h1=360, h2=360, out_features=21):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

torch.manual_seed(42)
model = Model()

X = pd.read_csv('traindata.txt', header=None)
Y = pd.read_csv('trainlabels.txt', header=None)

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 100
errors = []

for i in range(epochs):
    Y_pred = model.forward(X_train)
    error = criterion(Y_pred, Y_train.squeeze())
    errors.append(error)

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {error}')

    optimizer.zero_grad()
    error.backward()
    optimizer.step()

correct = 0
wrong = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        if y_val.argmax().item() == Y_test[i].item():
            correct += 1
        else:
            wrong += 1
print(f'Correct: {correct}, Wrong: {wrong}')

torch.save(model.state_dict(), 'model.pth')