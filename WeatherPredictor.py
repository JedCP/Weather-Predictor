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

def import_and_predict(image_data, model):
    if model is None:
        return None
    
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    try:
        prediction = model.predict(img_array)
        class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.title("Weather Predictor")
    st.write("Upload an image to classify the weather conditions.")
    
    model = load_model()
    
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        if st.button("Predict"):
            with st.spinner('Predicting...'):
                prediction = import_and_predict(image, model)
                if prediction is not None:
                    st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
