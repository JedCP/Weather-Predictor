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
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("Weather Predictor")
    st.write("Upload an image to classify the weather conditions.")
    
    model = load_model()
    
    if model is None:
        st.stop()
    
    st.write("## Upload an Image")
    st.write("Drag and drop an image file below, or click to select a file.")
    
    file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        with st.spinner('Processing Image...'):
            prediction = import_and_predict(image, model)
            class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = class_labels[predicted_class_index]

            st.success(f'Prediction: {predicted_class_label}')

if __name__ == '__main__':
    main()
