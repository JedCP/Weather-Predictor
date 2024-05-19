import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Weather Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model with caching
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model("model_Weather.h5")
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
if model is None:
        st.error("Model is not loaded.")
        return None

    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    prediction = model.predict(img_array)  # Example prediction

    return prediction

def main():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                prediction = import_and_predict(image, model)
                st.success(f'Prediction: {prediction}')
