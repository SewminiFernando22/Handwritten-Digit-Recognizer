import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Digit Wizard", page_icon="✍️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title { font-family: 'Arial', sans-serif; color: #2c3e50; }
    .subheader { color: #34495e; }
    .success { color: #27ae60; font-weight: bold; }
    .stButton>button { background-color: #3498db; color: white; border-radius: 10px; }
    .stButton>button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'> Digit Wizard </h1> <br><h2> Handwritten Digit Recognizer</h2>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload an image of a handwritten digit (0-9) and let our AI predict it! Draw a digit in black on a white background for best results.</p>", unsafe_allow_html=True)

# Sidebar for controls and info
with st.sidebar:
    st.header("About This App")
    st.write("Built with a Convolutional Neural Network (CNN) trained on the MNIST dataset, achieving ~98% accuracy. Upload a PNG/JPG image or try a sample digit!")
    st.image("https://storage.googleapis.com/kaggle-datasets-images/1395/2487/6b3c76b2c31dd5e4a2bafa0935e0d3f7/dataset-card.jpg?t=2018-03-22-18-37-24", caption="MNIST Sample", use_column_width=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)

# Load the model
try:
    model = load_model('digit_recognizer.h5')
    st.success("Model loaded successfully! Ready to predict digits.")
except FileNotFoundError:
    st.error("Error: 'digit_recognizer.h5' not found. Please train and save the model in your Jupyter notebook.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a handwritten digit (PNG/JPG)", type=["png", "jpg"], help="Draw a digit (0-9) in black on a white background, ideally ~28x28 pixels.")

if uploaded_file:
    try:
        # Process the image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)
        img_resized = cv2.resize(img, (28, 28)) / 255.0
        img_input = img_resized.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_input)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display image and prediction
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_resized, caption="Processed Image (28x28)", use_column_width=True)
        with col2:
            st.markdown(f"<p class='success'>Predicted Digit: {predicted_digit}</p>", unsafe_allow_html=True)
            st.write(f"Confidence: {confidence:.2f}%")

        # Show confidence scores as a bar chart (if enabled)
        if show_confidence:
            fig, ax = plt.subplots()
            ax.bar(range(10), prediction[0], color='#3498db')
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}. Ensure the image is valid and try again.")

# Footer
st.markdown("<hr><p style='text-align: center; color: #7f8c8d;'>Created with ❤️ using Streamlit & TensorFlow</p>", unsafe_allow_html=True)