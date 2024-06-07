import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Preprocess the data
def preprocess_data(data):
    data[data < 0] = 0  # Remove negative features by setting them to zero
    data = data / 255.0  # Normalize values to [0, 1]
    return data

def rotate_image(image, orientation):
    if orientation == 0:
        return image  # Original orientation
    elif orientation == 1:
        return np.rot90(image, k=1, axes=(1, 2))  # 90 degrees clockwise
    elif orientation == 2:
        return np.rot90(image, k=2, axes=(1, 2))  # 180 degrees
    elif orientation == 3:
        return np.rot90(image, k=3, axes=(1, 2))  # 270 degrees (or 90 degrees counterclockwise)
    return image


# Read the data
df_features = pd.read_csv('traindata.txt', delimiter=',', header=None)
df_labels = pd.read_csv('trainlabels.txt', header=None)

# Separate orientation from features
orientations = df_features.iloc[:, -1].to_numpy()
df_features = df_features.iloc[:, :-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, orientations_train, orientations_test = train_test_split(
    df_features,
    df_labels,
    orientations,
    test_size=0.3,
    random_state=42
)

# Convert the data to numpy arrays and preprocess
X_train = preprocess_data(X_train.to_numpy())
X_test = preprocess_data(X_test.to_numpy())
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Determine the correct image shape
rows, cols = 40, 26  # Correct dimensions

# Reshape the data into image blocks and add channel dimension
X_train = X_train.reshape(-1, 1, rows, cols)
X_test = X_test.reshape(-1, 1, rows, cols)

X_train_rotated_list = [rotate_image(img, ori) for img, ori in zip(X_train, orientations_train)]
X_test_rotated_list = [rotate_image(img, ori) for img, ori in zip(X_test, orientations_test)]

# Convert lists to numpy arrays
X_train_rotated = np.array(X_train_rotated_list)
X_test_rotated = np.array(X_test_rotated_list)


# Convert data to PyTorch tensors
X_train_rotated = torch.tensor(X_train_rotated, dtype=torch.float32)
X_test_rotated = torch.tensor(X_test_rotated, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the CNN model using PyTorch
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channel is 1 for grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 10 * 6, 128),  # Update this if the input shape changes
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Define constants
NUM_CLASSES = len(np.unique(y_train))
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 64

# Initialize the model, loss function, and optimizer
model = CNN(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for i in range(0, X_train.size(0), BATCH_SIZE):
        optimizer.zero_grad()
        outputs = model(X_train[i:i+BATCH_SIZE])
        loss = criterion(outputs, y_train[i:i+BATCH_SIZE])
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_test_pred_prob = model(X_test)
    y_test_pred = torch.argmax(y_test_pred_prob, axis=1)
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Model accuracy: {:.2%}'.format(accuracy))

# Save the model
torch.save(model.state_dict(), 'model.pth')