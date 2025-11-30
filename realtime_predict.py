# realtime_predict.py
import cv2
import numpy as np
import tensorflow as tf
import argparse
import time

def load_model(path):
    # prefer saved_model if available, fallback to h5
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print("Failed loading model:", e)
        raise

def main(args):
    model_path = args.model
    img_size = args.size
    model = load_model(model_path)
    class_names = args.classes.split(",") if args.classes else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    box = args.box
    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box // 2, cy - box // 2
        x2, y2 = cx + box // 2, cy + box // 2

        roi = frame[y1:y2, x1:x2]
        img = cv2.resize(roi, (img_size, img_size))
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img.astype("float32"), axis=0))

        preds = model.predict(img_arr)
        top_idx = np.argmax(preds[0])
        conf = preds[0][top_idx]
        label = class_names[top_idx] if class_names else str(top_idx)

        # Draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        cv2.imshow("Visual Voice - Live Prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gesture_model.h5", help="Path to trained model (.h5 or SavedModel dir)")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated class names in index order (optional). If omitted, class order printed during training must be used.")
    parser.add_argument("--size", type=int, default=224, help="Input image size used for training")
    parser.add_argument("--box", type=int, default=300, help="ROI box size")
    args = parser.parse_args()
    main(args)
