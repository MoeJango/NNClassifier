import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import numpy as np
data = pd.read_csv('traindata.txt', header=None).values



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
    
def normalize_pixel_values(data, max_value=2550):
    """
    Normalize pixel values to be between 0 and 255.
    """
    # Shift data so minimum value is 0
    data = data - data.min()
    # Scale data to the range 0-255
    data = (data / max_value) * 255
    return data

def shape_data(data):
    for d, o in zip(data, orientations):
        if o == 1 or o == 3:
            d = d.reshape(26, 40)
        else:
            d = d.reshape(40, 26)
    for img, ori in zip(data, orientations):
        rotate_image(img, ori)
    return data

# Remove the last column (orientation)
orientations = data[:, -1]
data = data[:, :-1]

# Normalize the data
data = normalize_pixel_values(data, max_value=2550)


images = shape_data(data)




