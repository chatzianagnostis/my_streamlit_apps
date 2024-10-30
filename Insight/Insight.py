import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from utils import preprocess_image, predict, visualize_prediction

# Set directory where models are stored
model_dir = './models/'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
model_options = [os.path.join(model_dir, f) for f in model_files]

# File uploader for image input
st.title("Image Segmentation with Multiple Models")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_image is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as a color image (BGR format)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    # Preprocess image
    #st.write("### Preprocessing Image...")
    preprocessed_image = preprocess_image(image)

# Initialize session state for slide index and predictions
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Function to increment slide index
def next_slide():
    st.session_state.slide_index = (st.session_state.slide_index + 1) % len(st.session_state.predictions)

# Function to decrement slide index
def prev_slide():
    st.session_state.slide_index = (st.session_state.slide_index - 1) % len(st.session_state.predictions)

# Prediction and visualization section
if st.button("Predict with All Models"):
    predictions = {}  # Temporary dictionary to store predictions

    # Loop through each model and generate predictions
    for model_path in model_options:
        model_name = os.path.basename(model_path)
        #st.write(f"### Making Predictions with model: {model_name}...")

        # Predict and visualize using the selected model
        prediction = predict(model_path, preprocessed_image)
        visualization = visualize_prediction(image, prediction)
        predictions[model_name] = visualization

    # Save predictions in session state
    st.session_state.predictions = predictions
    st.session_state.slide_index = 0  # Reset to first image

# Display section - only if predictions are available
if st.session_state.predictions:
    # Get list of model names and corresponding images
    model_names = list(st.session_state.predictions.keys())
    images = list(st.session_state.predictions.values())

    # Display the current image based on the slide index
    current_model_name = model_names[st.session_state.slide_index]
    current_image = images[st.session_state.slide_index]

    # Show the current image with its model name
    st.image(current_image, caption=current_model_name, use_column_width=True)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("Previous"):
            prev_slide()
    with col3:
        if st.button("Next"):
            next_slide()