import os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Sruns/segment/train/weights/best.pt",
                    help="path to trained *.pt")
    ap.add_argument("--data",  default="data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf",  type=float, default=0.001, help="score threshold used for NMS during eval")
    ap.add_argument("--iou",   type=float, default=0.7,   help="IoU threshold for mAP calc")
    ap.add_argument("--split", default="val", choices=["val","test"])
    args = ap.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,   # uses test set if your data.yaml has test:
        conf=args.conf,
        iou=args.iou,
        plots=True,         # PR curves, confusion, etc. in runs dir
        project="Sruns/segment",
        name=f"val-{args.split}",
        exist_ok=True
    )

    # Key segmentation numbers:
    # Ultralytics exposes .seg for mask AP (COCO-style box/mask metrics are separate)
    print("Seg mAP50-95:", metrics.seg.map)
    print("Seg mAP50   :", metrics.seg.map50)
    print("Seg mAP75   :", metrics.seg.map75)
    print("Per-class Seg mAP:", metrics.seg.maps)

if __name__ == "__main__":
    main()
