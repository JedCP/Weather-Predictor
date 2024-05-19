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
    
    model = load_model()
    
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        with st.spinner('Processing Image...'):
            # Preprocess the image
            image = ImageOps.fit(image, (150, 150), Image.LANCZOS)
            img_array = np.asarray(image)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the class of the image
            try:
                with st.spinner('Classifying...'):
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]

                    st.success("Classification Successful!")
                    st.write(f"Predicted Class: {predicted_class}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
