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
    def __init__(self, in_features=1040, h1=360, h2=360, out_features=21):
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


def normalize_pixel_values(data, max_value, min_value):
    """
    Normalize pixel values to be between 0 and 255.
    """
    # Shift data so minimum value is 0
    data = data - min_value
    # Scale data to the range 0-255
    data = (data / max_value) * 255
    return data

def rotate_image(image, orientation):
    """
    Rotate image to the normal orientation.
    """
    if orientation == 1:  # 90 degrees
        return np.rot90(image, 3)
    elif orientation == 2:  # 180 degrees
        return np.rot90(image, 2)
    elif orientation == 3:  # 270 degrees
        return np.rot90(image, 1)
    else:
        return image



def shape_data(data, orientations):
    shaped_data = []
    for d, o in zip(data, orientations):
        if o == 1 or o == 3:
            d = d.reshape(26, 40)
        else:
            d = d.reshape(40, 26)
        shaped_data.append(d)
    shaped_data_rotated = []
    for img, ori in zip(shaped_data, orientations):
        img = rotate_image(img, ori)
        shaped_data_rotated.append(img.flatten())
    return np.array(shaped_data_rotated)

X = pd.read_csv('testdata.txt', header=None)
X_test = X.values
orientations = X_test[:, -1]
X_test = X_test[:, :-1]

min_value = np.min(X_test)
max_value = np.max(X_test)

X_test = normalize_pixel_values(X_test, max_value, min_value)
X_test = shape_data(X_test, orientations)
StandardScaler().fit_transform(X_test)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

with open('testlabels.txt', 'a') as f:
    for point in X_test:
        point = torch.FloatTensor(point)
        pred = model.forward(point)
        pred = torch.argmax(pred).item()
        f.write(f'{pred}\n')


