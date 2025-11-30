⭐ Visual Voice – AI-Powered Sign Language Detection
Real-time gesture recognition using deep learning, computer vision, and a fully interactive web interface.

📌 Overview

Visual Voice is an end-to-end AI system that detects sign language gestures using:

📷 Real-time webcam inference

🖼️ Image upload prediction

🎥 Automatic dataset collection via webcam

🧠 Custom gesture training with CNN

🌐 Beautiful Flask-based UI

It allows anyone to create, collect, train, and test custom gesture datasets—no ML expertise required.

🚀 Features
🔹 1. Real-Time Gesture Detection

Uses your webcam to predict gestures live with confidence scores.

🔹 2. Upload & Predict

Upload a static gesture image and get instant predictions.

🔹 3. Automatic Dataset Collection

Collect gesture dataset using webcam with:

Auto-capture mode

Live ROI box

Organized dataset folders

🔹 4. Train Your Own Model

Train a CNN with your custom dataset using a single click.

🔹 5. View Last Prediction

Displays the most recent result from the prediction module.

🔹 6. Clean Modern UI

Bootstrap-powered card layout:

Live Detection

Upload Dataset

Upload Image

Train Model

View Last Prediction

🔹 7. Extendable Architecture

Add new gestures anytime—no rewriting required.

📂 Project Structure
VisualVoice/
│── app.py
│── collect_gestures.py
│── model/
│── dataset/
│── static/
│   ├── upload.jpg
│   └── styles.css
│── templates/
│   ├── index.html
│   ├── live.html
│   ├── result.html
│   ├── upload_dataset.html
│   └── static_prediction.html
│── README.md

🧠 Model

A lightweight CNN designed for speed and realtime accuracy:

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

Flatten

Dense(128)

Dense(#labels) + Softmax

Trains in seconds on CPU.

🛠️ Installation & Setup
Clone Repository
git clone https://github.com/YourUsername/VisualVoice.git
cd VisualVoice

Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Run App
python app.py

🧪 Usage Workflow
1️⃣ Add a New Gesture

Go to → Upload New Gesture Dataset
→ Enter label name
→ Webcam opens and captures images automatically

2️⃣ Train Model

Click Train Model
→ Model is saved to model/gesture_model.h5

3️⃣ Predict

Use either:

Live Detection

Upload Image

4️⃣ View Last Prediction

Shows the most recent static prediction image & result.
<img width="1875" height="919" alt="Screenshot 2025-12-01 020136" src="https://github.com/user-attachments/assets/0842bf23-2865-4fd5-a6df-9aebd5d6902d" />

<img width="975" height="699" alt="Screenshot 2025-12-01 020147" src="https://github.com/user-attachments/assets/ee6f55b9-5a9a-40e4-af6d-cc7f994f01a3" />
<img width="838" height="641" alt="Screenshot 2025-12-01 020326" src="https://github.com/user-attachments/assets/17ac4579-08ef-4a2d-babd-fbed450a0762" />
<img width="579" height="780" alt="Screenshot 2025-12-01 021411" src="https://github.com/user-attachments/assets/498dca70-6489-442b-82b9-dc720ef09dad" />




🔮 Future Upgrades (Recommended Features)

Here are some great enhancements you can add:

📌 AI Features

Transformer-based gesture recognition

3D hand landmark detection (MediaPipe)

Multi-gesture sentences using sequence models

American Sign Language (ASL) alphabet mode

📌 UX / UI Features

Dark/Light theme switch

Dashboard analytics for dataset size

Gesture preview gallery

📌 Developer Features

REST API for gesture inference

WebSocket real-time streaming

Export model to ONNX / TFLite

Let me know—I can implement any of these.

🤝 Contributing

Pull requests are welcome.
For major changes, open an issue first to discuss what you'd like to improve.

📝 License

MIT License.

💡 Credits

Created by Dhruv
Built with ❤️ using Python, Flask, TensorFlow & OpenCV.
