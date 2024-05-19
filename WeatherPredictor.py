import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set page configuration
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Weather Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS styling
st.markdown(
    
    <style>
    /* Add your custom CSS styling here */
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }

    .stApp {
        max-width: 1.1px;
        padding: 1rem;
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        border-radius: 5px;
        padding: 1rem;
    }

    .sidebar .sidebar-content .block-container {
        margin-bottom: 1rem;
    }

    .block-container label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .block-container input[type="text"],
    .block-container input[type="number"],
    .block-container select {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #cccccc;
        border-radius: 3px;
        background-color: #ffffff;
    }

    .block-container input[type="text"]:focus,
    .block-container input[type="number"]:focus,
    .block-container select:focus {
        outline: none;
        border-color: #0071e3;
        box-shadow: 0 0 5px rgba(0, 113, 227, 0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load model with caching
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model("Weather_Predictor.h5")
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("Weather Predictor")
st.write("Upload an image to classify the weather conditions.")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    prediction = model.predict(img_array)  # Example prediction
    return prediction

if uploaded_file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    st.success(f'Prediction: {predicted_class_label}')
