import os, argparse, cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--source", required=True, help="Path to image")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    model = YOLO(args.model)
    img = cv2.imread(args.source)
    results = model.predict(img, conf=args.conf, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
