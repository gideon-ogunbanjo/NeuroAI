# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading the data
data = pd.read_csv('./Data/train.csv')

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

''' The Neural Network will have a simple two-layer architecture. Input layer - 𝑎[0], will have 784 units corresponding to the 784 
pixels in each 28x28 input image. A hidden layer - 𝑎[1], will have 10 units with ReLU activation, and finally our output layer - 𝑎[2], 
will have 10 units corresponding to the ten digit classes with softmax activation.
'''
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

''' Using gradient descent for optimization and printing the accuracy of the NeuroAI's 
predictions at every 10 iterations during training.
'''
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Running Predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
# Running Tests
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)