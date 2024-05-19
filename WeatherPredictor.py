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
    model = tf.keras.models.load_model("model_Weather.h5")
    return model

# Main function
def main():
    st.title("Weather Predictor")
    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = ImageOps.fit(image, (150, 150), Image.ANTIALIAS)
        img_array = np.asarray(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class of the image
        predictions = model.predict(img_array)
        st.write(predictions)

if _name_ == '_main_':
    main()
