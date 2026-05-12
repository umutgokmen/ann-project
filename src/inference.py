"""
Single-image and webcam inference for Car Model Recognition.

Usage:
  python src/inference.py --image path/to/car.jpg
  python src/inference.py --webcam
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import CarClassifier, load_checkpoint


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CarRecognizer:
    def __init__(self, checkpoint_path: str, class_names_path: str, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        with open(class_names_path) as f:
            self.class_names = json.load(f)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["cfg"]

        self.model = CarClassifier(
            backbone=cfg["model"]["backbone"],
            num_classes=cfg["data"]["num_classes"],
            dropout=0.0,
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = build_inference_transform(cfg["data"]["image_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(top_k)
        results = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            results.append({"class": self.class_names[idx], "confidence": prob})
        return results

    def predict_from_path(self, image_path: str, top_k: int = 5) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        return self.predict(image, top_k)


def predict_image(recognizer: CarRecognizer, image_path: str):
    results = recognizer.predict_from_path(image_path)
    print(f"\nImage: {image_path}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class']:<45} {r['confidence'] * 100:.1f}%")


def run_webcam(recognizer: CarRecognizer, top_k: int = 5):
    try:
        import cv2
    except ImportError:
        print("OpenCV required for webcam: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to quit, 's' to capture and predict.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 's' to predict, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Car Recognizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            results = recognizer.predict(image, top_k)

            print("\n--- Prediction ---")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['class']:<45} {r['confidence'] * 100:.1f}%")

            # Overlay top result on frame
            label = f"{results[0]['class']} ({results[0]['confidence'] * 100:.1f}%)"
            cv2.putText(frame, label, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Car Recognizer", frame)
            cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--class_names", default="checkpoints/class_names.json")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    recognizer = CarRecognizer(args.checkpoint, args.class_names, args.device)
    print(f"Model loaded on {recognizer.device}")

    if args.image:
        predict_image(recognizer, args.image)
    elif args.webcam:
        run_webcam(recognizer, args.top_k)
    else:
        print("Provide --image <path> or --webcam")


if __name__ == "__main__":
    main()
