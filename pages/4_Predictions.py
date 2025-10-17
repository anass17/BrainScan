import streamlit as st
from tensorflow.keras.models import load_model
import os
from PIL import Image
from helpers import predict_image

dir = os.getcwd() + "/models/"

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

best_model = load_model(dir + 'model.keras')

if uploaded_file:
    img = Image.open(uploaded_file)

    res = predict_image(best_model, img)

    st.success(res[0])

    st.image(img, caption="Uploaded Image")
