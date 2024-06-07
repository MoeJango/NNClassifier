import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from scipy.stats import zscore
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features=1040, h1=300, h2=300, out_features=21):
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

# Remove the last column (orientation)
orientations_train = X_train[:, -1]
X_train = X_train[:, :-1]
orientations_test = X_test[:, -1]
X_test = X_test[:, :-1]

max_value = np.max(X.values)
min_value = np.min(X.values)
# Normalize the data
X_train = normalize_pixel_values(X_train, max_value, min_value)
X_test = normalize_pixel_values(X_test, max_value, min_value)

X_train = shape_data(X_train, orientations_train)
X_test = shape_data(X_test, orientations_test)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 150
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