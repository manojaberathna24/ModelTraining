import os
import joblib
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
import sys

# Use absolute path or relative path from app folder
model_path = "C:/Users/Manoj Aberathna/Desktop/car_damage_classifier/model/car_damage_svm.pkl"
model = joblib.load(model_path)

img_path = sys.argv[1]
img = imread(img_path)
img = resize(img, (150,150))

features, _ = hog(img, orientations=9, pixels_per_cell=(16,16),
                  cells_per_block=(2,2), visualize=True, channel_axis=-1)
features = np.array(features).reshape(1, -1)

pred = model.predict(features)[0]

if pred == 0:
    print("Dent Detected!")
else:
    print("Scratch Detected!")
