from flask import Flask, render_template, request, Response
import tensorflow as tf
import numpy as np
import cv2
import os
import subprocess
import sys

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "dataset"

MODEL_PATH = "model/gesture_model.h5"
IMG_SIZE = 64

# Store last prediction globally
LAST_PREDICTION = {
    "label": None,
    "confidence": None,
    "img_path": None
}

# Load model if exists
MODEL = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Labels
GESTURES = sorted(os.listdir("dataset")) if os.path.exists("dataset") else []

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# LIVE DETECTION PAGE
# -------------------------------
@app.route("/live")
def live():
    return render_template("live.html")

# -------------------------------
# VIDEO STREAM GENERATOR
# -------------------------------
def generate_frames():
    global MODEL
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        if MODEL is not None and GESTURES:
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = MODEL.predict(img, verbose=0)[0]
            label = GESTURES[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100

            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------------------
# PREDICT FROM UPLOADED IMAGE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global MODEL, GESTURES, LAST_PREDICTION

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    path = "static/upload.jpg"
    file.save(path)

    if MODEL is None or not GESTURES:
        return "Model not trained yet!"

    img = cv2.imread(path)
    processed = preprocess(img)

    pred = MODEL.predict(processed)[0]
    label = GESTURES[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 2)

    # Save last prediction details
    LAST_PREDICTION["label"] = label
    LAST_PREDICTION["confidence"] = confidence
    LAST_PREDICTION["img_path"] = path

    return render_template("result.html", label=label, confidence=confidence, img_path=path)

# -------------------------------
# VIEW LAST STATIC PREDICTION
# -------------------------------
@app.route("/static_prediction")
def static_prediction():
    if LAST_PREDICTION["label"] is None:
        return "No prediction has been made yet!"

    return render_template(
        "result.html",
        label=LAST_PREDICTION["label"],
        confidence=LAST_PREDICTION["confidence"],
        img_path=LAST_PREDICTION["img_path"]
    )

# -------------------------------
# UPLOAD NEW LABEL (Triggers Webcam Multi-Capture)
# -------------------------------
@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    global GESTURES

    if request.method == "POST":
        label = request.form.get("label", "").strip()
        if not label:
            return "Please provide a label name!"

        try:
            subprocess.run([sys.executable, "collect_gestures.py",
                            "--label", label,
                            "--dataset", "dataset",
                            "--auto",
                            "--interval", "5"], check=True)
        except subprocess.CalledProcessError as e:
            print("Error running collect_gestures.py:", e)
            return "Error capturing gestures from webcam."

        GESTURES = sorted(os.listdir("dataset"))

        return f"Webcam capture completed for label '{label}'!"

    return render_template("upload_dataset.html")

# -------------------------------
# TRAIN MODEL ROUTE
# -------------------------------
@app.route("/train_model")
def train_model():
    global MODEL, GESTURES

    if not os.path.exists("dataset"):
        return "No dataset available!"

    print("Training model...")
    data = []
    labels = []

    GESTURES = sorted(os.listdir("dataset"))

    for idx, gesture in enumerate(GESTURES):
        folder = os.path.join("dataset", gesture)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)

    if not data:
        return "Dataset is empty!"

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(GESTURES), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(data, labels, epochs=7)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    MODEL = tf.keras.models.load_model(MODEL_PATH)

    return "Model trained successfully!"

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
