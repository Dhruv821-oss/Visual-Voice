# collect_gestures.py
import cv2
import os
import argparse
from datetime import datetime
import sys

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_from_webcam(label, dataset_dir, box_size=300, auto=False, interval=5):
    out_dir = os.path.join(dataset_dir, label)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    print("Starting capture. Press SPACE to capture manually. Press 'a' to toggle auto-capture. Press 'q' to quit.")
    print(f"Saving images to: {out_dir}")
    count = len(os.listdir(out_dir))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        # Draw ROI rectangle
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = cx + box_size // 2, cy + box_size // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Label: {label} Count: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Auto: {'ON' if auto else 'OFF'} (toggle 'a')", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Collect Gestures", frame)

        key = cv2.waitKey(1) & 0xFF
        do_capture = False

        if key == ord(' '):  # manual capture
            do_capture = True
        elif key == ord('a'):
            auto = not auto
            print("Auto-capture:", auto)
        elif key == ord('q'):
            break

        if auto and frame_idx % interval == 0:
            do_capture = True

        if do_capture:
            roi = frame[y1:y2, x1:x2]
            filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, roi)
            count += 1
            print("Saved:", path)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def add_image_to_dataset(image_path, label, dataset_dir="dataset"):
    """
    Add a single uploaded image to the dataset folder.
    This is useful for Flask integration.
    """
    out_dir = os.path.join(dataset_dir, label)
    ensure_dir(out_dir)

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image:", image_path)
        return

    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print("Added uploaded image to dataset:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hand gesture images for a single label.")
    parser.add_argument("--label", type=str, required=True, help="Label name (folder under dataset/)")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset folder")
    parser.add_argument("--box", type=int, default=300, help="ROI box size (square)")
    parser.add_argument("--auto", action="store_true", help="Start in auto-capture mode")
    parser.add_argument("--interval", type=int, default=5, help="Auto-capture every N frames")
    parser.add_argument("--image", type=str, help="Optional: add single image directly to dataset (for Flask)")
    args = parser.parse_args()

    if args.image:
        add_image_to_dataset(args.image, args.label, args.dataset)
    else:
        capture_from_webcam(args.label, args.dataset, args.box, args.auto, args.interval)
