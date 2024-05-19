import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

st.markdown(
    
    <style>
    body {
        background-image: url('path_to_your_image.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
# Set page configuration
st.set_page_config(
    page_title="Weather Classification Proficiency Test",
    layout="centered",
    initial_sidebar_state="expanded"
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

st.title("Weather Classification Proficiency Test")
st.write("Pernecita, Jed Carlo | Garcia, Christian")
st.write("Final Exam")
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
