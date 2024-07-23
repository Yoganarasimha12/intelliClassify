import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

model_path = 'models/trained_model.h5'
model = tf.keras.models.load_model(model_path)

class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def main():
    st.title("Satellite Image Classification")

    uploaded_files = st.file_uploader("Choose satellite images...", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Number of images uploaded: {len(uploaded_files)}")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            preprocessed_image = preprocess_image(image, target_size=(64, 64))

            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            st.write(f"Prediction: {class_names[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
