import streamlit as st
import numpy as np
from PIL import Image
import joblib

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("iris_model.pkl")
species = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Classifier")
st.write("Upload a flower image and I will predict if it is Setosa, Versicolor, or Virginica.")

# ---------------------------
# Image Upload
# ---------------------------
uploaded_img = st.file_uploader("Upload Flower Image", type=["jpg", "png", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------------------
    # Feature Extraction From Image
    # -----------------------------------------
    img = img.resize((150, 150))
    img_np = np.array(img)

    R_mean = np.mean(img_np[:, :, 0])
    G_mean = np.mean(img_np[:, :, 1])
    B_mean = np.mean(img_np[:, :, 2])
    brightness = np.mean(img_np)

    # Map extracted features to iris model as a placeholder
    # (Since iris has 4 numeric features)
    features = np.array([[R_mean/255, G_mean/255, B_mean/255, brightness/255]])

    # -----------------------------------------
    # Predict
    # -----------------------------------------
    prediction = model.predict(features)[0]
    predicted_species = species[prediction]

    st.subheader("🌼 Predicted Species:")
    st.success(predicted_species)