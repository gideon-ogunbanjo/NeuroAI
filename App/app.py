import streamlit as st
import numpy as np
from neural_network import predict_digit
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NeuroAI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Function to convert the image to grayscale and reshape it to (28, 28)
def preprocess_image(image):
    img_gray = image.convert('L')
    img_resize = img_gray.resize((28, 28))
    img_array = np.array(img_resize)
    return img_array

def main():
    st.title("Neural Network Digit Recognition")
    st.write("Draw a digit from 0 to 9 and click 'Predict' to see the prediction.")
    canvas = st.canvas(draw_text='Draw here:', height=150)

    if st.button("Predict"):
        # Get the canvas image and preprocess it
        img_data = canvas.image_data.astype(np.uint8)
        img = Image.fromarray(img_data)
        img_array = preprocess_image(img)

        # Predict the digit using the neural network
        prediction = predict_digit(img_array, W1, b1, W2, b2)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    # Load the neural network weights
    # Replace the paths with the actual paths to your trained weights
    W1 = np.load("path_to_W1.npy")
    b1 = np.load("path_to_b1.npy")
    W2 = np.load("path_to_W2.npy")
    b2 = np.load("path_to_b2.npy")

    main()
# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)