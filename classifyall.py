import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
model_path = "model.pth"
model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

X = pd.read_csv('testdata.txt', header=None)
X_test = X.values
pca = PCA(n_components=100)
X_test = pca.fit_transform(X_test)

X_test = X_test + np.random.normal(0, 0.1, size=X_test.shape)
StandardScaler().fit_transform(X_test)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

with open('testlabels.txt', 'a') as f:
    for point in X_test:
        point = torch.FloatTensor(point)
        pred = model.forward(point)
        pred = torch.argmax(pred).item()
        f.write(f'{pred}\n')


