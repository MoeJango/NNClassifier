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

# Rotate Data Function
def rotate_data(data, angle):
    rotated_data = np.concatenate((data[:, angle:], data[:, :angle]), axis=1)
    return rotated_data

# Add Noise Function
def add_noise(data, mean, std_dev):
    noisy_data = data + np.random.normal(mean, std_dev, size=data.shape)
    return noisy_data

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
num_features = X_train.shape[1]
image_height = 32  # Assuming a height of 32 pixels
image_width = num_features // image_height  # Calculate width based on the number of features
image_shape = (1, image_height, image_width)  # (channels, height, width)


# Reshape the data into image blocks
X_train = X_train.reshape((-1, *image_shape))
X_test = X_test.reshape((-1, *image_shape))

# Data augmentation - random perturbations
augmented_X_train = []
augmented_y_train = []

for i in range(len(X_train)):
    original_data = X_train[i]
    augmented_X_train.append(original_data.reshape(image_shape))
    augmented_y_train.append(y_train[i])

    # Apply random perturbations
    perturbed_data = original_data + np.random.normal(0, 0.1, size=image_shape)
    augmented_X_train.append(perturbed_data)
    augmented_y_train.append(y_train[i])

# Convert augmented data to numpy arrays
augmented_X_train = np.array(augmented_X_train)
augmented_y_train = np.array(augmented_y_train)

# Concatenate augmented data with original data
X_train = np.concatenate([X_train.reshape((-1, *image_shape)), augmented_X_train], axis=0)
y_train = np.concatenate([y_train, augmented_y_train], axis=0)

# Shuffle the augmented data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Rotate the data
X_train = rotate_data(X_train, angle=10)
X_test = rotate_data(X_test, angle=10)

# Add noise to the data
X_train = add_noise(X_train, 0, 0.1)
X_test = add_noise(X_test, 0, 0.1)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
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
            nn.Linear(64 * 8 * 8, 128),  # Update this if the input shape changes
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
