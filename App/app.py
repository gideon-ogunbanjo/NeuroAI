import streamlit as st
import numpy as np
from neural_network import predict_digit
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
    st.title("NeuroAI - Neural Network Digit Recognition")
    st.write("Draw a digit from 0 to 9 and click 'Predict' to see the prediction.")
    
    # Use st_canvas to create the drawing canvas for drawing the digit
    canvas_result = st_canvas(
        fill_color="#000000",  # Background color of the canvas
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Get the canvas image and preprocess it
            img_data = np.array(canvas_result.image_data).astype(np.uint8)
            img = Image.fromarray(img_data)
            img_array = preprocess_image(img)

            # Load the neural network weights
            # Replace the paths with the actual paths to your trained weights
            W1 = np.load("path_to_W1.npy")
            b1 = np.load("path_to_b1.npy")
            W2 = np.load("path_to_W2.npy")
            b2 = np.load("path_to_b2.npy")

            # Predict the digit using the neural network
            prediction = predict_digit(img_array, W1, b1, W2, b2)
            st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)