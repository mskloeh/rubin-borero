import os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="yolo11n-seg.pt", help="yolo11[n|s|m|l|x]-seg.pt")
    ap.add_argument("--data",    default="data.yaml",      help="YOLO data.yaml path")
    ap.add_argument("--epochs",  type=int, default=100)
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--batch",   type=int, default=16)
    ap.add_argument("--device",  default=None, help="0 for GPU0, 'cpu' for CPU, etc.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="Sruns/segment", help="Ultralytics project dir")
    ap.add_argument("--name",    default="train",         help="Run name")
    args = ap.parse_args()

    # Using a *segmentation* model selects the segmentation task automatically.
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=50,   # early stop
        cos_lr=True,   # cosine LR
        amp=True       # mixed precision if available
    )

if __name__ == "__main__":
    main()
