import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Weather-Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache.resource
def load_model():
    model = tf.keras.models.load_model("model_Weather.h5")
    return model
