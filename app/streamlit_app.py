import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import streamlit as st
import joblib

# Load trained model
model = joblib.load("model/car_damage_svm.pkl")  # adjust path if needed

st.title("Car Damage Detection")
st.write("Upload an image of your car to identify Dent or Scratch")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button("Identify Damage"):
        img = imread(uploaded_file)
        img = resize(img, (150,150))
        
        features, _ = hog(img, orientations=9, pixels_per_cell=(16,16),
                          cells_per_block=(2,2), visualize=True, channel_axis=-1)
        features = np.array(features).reshape(1, -1)
        
        pred = model.predict(features)[0]
        if pred == 0:
            st.success("Dent Detected!")
        else:
            st.success("Scratch Detected!")
