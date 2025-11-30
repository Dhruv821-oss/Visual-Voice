import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

DATASET_DIR = "dataset"
GESTURES = sorted(os.listdir(DATASET_DIR))

images = []
labels = []

IMG_SIZE = 64

# Load dataset
for idx, gesture in enumerate(GESTURES):
    folder = os.path.join(DATASET_DIR, gesture)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        images.append(img)
        labels.append(idx)

images = np.array(images)
labels = to_categorical(labels, num_classes=len(GESTURES))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(images, labels, epochs=10, batch_size=16)

os.makedirs("model", exist_ok=True)
model.save("model/gesture_model.h5")

print("MODEL SAVED SUCCESSFULLY!")
