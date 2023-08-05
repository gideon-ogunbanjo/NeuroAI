import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="NeuroAI",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# Read the data
data = pd.read_csv('./Data/train.csv')

data.head()
data.shape
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffling before splitting into dev and training sets

# Extract the first 1000 data samples from 'data' and transpose it to get 'data_dev'
data_dev = data[0:1000].T

# Separate labels (target variable) from 'data_dev' and store them in 'Y_dev'
Y_dev = data_dev[0]

# Extract features (input variables) from 'data_dev' (excluding the label column) and store them in 'X_dev'
X_dev = data_dev[1:n]

# Normalize the feature data in 'X_dev' by dividing each pixel value by 255 (scaling to 0-1 range)
X_dev = X_dev / 255.

# Extract the remaining data samples from 'data' (from 1001st sample to the end) and transpose it to get 'data_train'
data_train = data[1000:m].T

# Separate labels (target variable) from 'data_train' and store them in 'Y_train'
Y_train = data_train[0]

# Extract features (input variables) from 'data_train' (excluding the label column) and store them in 'X_train'
X_train = data_train[1:n]

# Normalize the feature data in 'X_train' by dividing each pixel value by 255 (scaling to 0-1 range)
X_train = X_train / 255.

# Define neural network functions
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

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
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

    # Save the neural network weights to numpy files
    # np.save("path_to_W1.npy", W1)
    # np.save("path_to_b1.npy", b1)
    # np.save("path_to_W2.npy", W2)
    # np.save("path_to_b2.npy", b2)

    return W1, b1, W2, b2


def predict_digit(image, W1, b1, W2, b2):
    image = image.reshape((-1, 1))
    image = image / 255.0
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)
    prediction = np.argmax(A2, axis=0)[0]
    return prediction

# Saves the neural network weights to numpy files
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# np.save("path_to_W1.npy", W1)
# np.save("path_to_b1.npy", b1)
# np.save("path_to_W2.npy", W2)
# np.save("path_to_b2.npy", b2)


# Defines the Streamlit app
def preprocess_digit(canvas_result):
    digit_image = canvas_result.image_data.astype('float32')
    digit_image = digit_image[0:28, 0:28]
    digit_image /= 255.0
    digit_image = digit_image.flatten()
    return digit_image.reshape((-1, 1))

def main():
    st.title("NeuroAI - Digit Classifier App")
    st.write("Draw a digit below and let NeuroAI predict it!")

    # Loads the neural network weights
    W1 = np.load("App/path_to_W1.npy")
    b1 = np.load("App/path_to_b1.npy")
    W2 = np.load("App/path_to_W2.npy")
    b2 = np.load("App/path_to_b2.npy")

    # Streamlit canvas to draw the digit
    canvas_result = st_canvas(
        fill_color="black",  # Background color for the canvas
        stroke_width=20,  # Stroke width of the drawing tool
        stroke_color="#FFFFFF",  # Stroke color of the drawing tool
        background_color="#000000",  # Background color of the canvas
        height=200,  # Height of the canvas
        width=200,  # Width of the canvas
        drawing_mode="freedraw",  # Drawing mode
        key="canvas",
    )

    # Adds a button to trigger prediction
    if st.button("Predict"):
        # Preprocess the drawn image
        digit_image = preprocess_digit(canvas_result)

        # Makes prediction using the trained model
        prediction = predict_digit(digit_image, W1, b1, W2, b2)

        # Show the predicted digit
        st.write(f"Predicted Digit: {prediction}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)