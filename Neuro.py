# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading the data
data = pd.read_csv('Data/train.csv')

data.head()
data.shape
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffling before splitting into dev and training sets

# Extracting the first 1000 data samples from 'data' and transpose it to get 'data_dev'.
data_dev = data[0:1000].T

# Separating labels (target variable) from 'data_dev' and store them in 'Y_dev'.
Y_dev = data_dev[0]

# Extracting features (input variables) from 'data_dev' (excluding the label column) and store them in 'X_dev'.
X_dev = data_dev[1:n]

# Normalizing the feature data in 'X_dev' by dividing each pixel value by 255 (scaling to 0-1 range).
X_dev = X_dev / 255.

# Extracting the remaining data samples from 'data' (from 1001st sample to the end) and transpose it to get 'data_train'.
data_train = data[1000:m].T

# Separating labels (target variable) from 'data_train' and store them in 'Y_train'.
Y_train = data_train[0]

# Extracting features (input variables) from 'data_train' (excluding the label column) and store them in 'X_train'.
X_train = data_train[1:n]

# Normalizing the feature data in 'X_train' by dividing each pixel value by 255 (scaling to 0-1 range).
X_train = X_train / 255.

# Note: The variable 'm_train' is computed but not used later in the code. It seems like a typo or incomplete part.
Y_train