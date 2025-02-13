import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load your trained model
model = tf.keras.models.load_model('MobileNet1_teeth_model_finetuned.keras')

# Class labels
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Streamlit app title
st.title("Teeth Image Classification")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of teeth", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    image_size = (128, 128)
    img = load_img(uploaded_file, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display the prediction
    st.write(f"### Predicted Class: {class_labels[predicted_class]}")
    st.write(f"### Confidence: {confidence:.2f}")
