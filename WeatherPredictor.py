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
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Weather Predictor")
    st.write("Upload an image to classify the weather conditions.")
    

    


            st.success(f'Prediction: {predicted_class_label}')

if __name__ == '__main__':
    main()
