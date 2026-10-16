import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_freshness_model.keras")
IMG_SIZE = (128, 128)

st.set_page_config(page_title="FreshHarvest Inspector", page_icon="🍎")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🍎 FreshHarvest Freshness Inspector")
st.write("Drag and drop an image of a banana, lemon, lulo, mango, orange, "
         "strawberry, tamarillo or tomato to check whether it is fresh or spoiled.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    arr = tf.image.resize(np.array(image), IMG_SIZE)
    arr = tf.expand_dims(arr, 0)

    p_spoiled = float(model.predict(arr, verbose=0)[0][0])

    if p_spoiled > 0.5:
        st.error(f"Prediction: SPOILED  (confidence {p_spoiled:.1%})")
    else:
        st.success(f"Prediction: FRESH  (confidence {1 - p_spoiled:.1%})")
