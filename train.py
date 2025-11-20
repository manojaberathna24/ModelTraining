import os
import joblib
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC

# Create model folder if not exist
if not os.path.exists("model"):
    os.makedirs("model")

data_dir = "dataset/"
categories = ["dent", "scratch"]

X = []
y = []

for label, cat in enumerate(categories):
    folder = os.path.join(data_dir, cat)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = imread(path)
        img = resize(img, (150,150))
        
        features, _ = hog(img, orientations=9, pixels_per_cell=(16,16),
                          cells_per_block=(2,2), visualize=True, channel_axis=-1)
        
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = SVC(kernel='linear', probability=True)
model.fit(X, y)

# Save model to model folder
model_path = os.path.join("model", "car_damage_svm.pkl")
joblib.dump(model, model_path)
print(f"Model saved as {model_path}")
