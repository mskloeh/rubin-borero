# detection_deploy.py
import os, time, argparse, cv2, requests, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ultralytics import YOLO

def iter_frames_cv2(src):
    print(f"[cv2] opening {src} ...")
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        cap.release()
        print("[cv2] no frame on open")
        return
    yield frame
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[cv2] stream ended")
            break
        yield frame
    cap.release()

def iter_frames_mjpeg(url, timeout=(5, 45)):
    print(f"[mjpeg] connecting to {url} ...")
    try:
        with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent":"YOLO/esp32"}) as r:
            r.raise_for_status()
            print(f"[mjpeg] {r.status_code} {r.headers.get('Content-Type','')}")
            buf = b""
            for chunk in r.iter_content(chunk_size=2048):
                if not chunk:
                    continue
                buf += chunk
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9")
                if a != -1 and b != -1 and b > a:
                    jpg = buf[a:b+2]
                    buf = buf[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame
    except Exception as e:
        print(f"[mjpeg] error: {e}")

def get_frame_generator(source, backend):
    if backend in ("cv2","auto"):
        gen = iter_frames_cv2(source)
        try:
            first = next(gen)
            yield first
            for f in gen: yield f
            if backend == "cv2":
                return
        except StopIteration:
            pass  # fall through if auto
    if isinstance(source, str) and source.startswith("http"):
        for f in iter_frames_mjpeg(source): yield f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to *.pt")
    ap.add_argument("--source", default="0", help="0 for webcam, or MJPEG URL")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)
    ap.add_argument("--backend", choices=["auto","cv2","mjpeg"], default="auto")
    ap.add_argument("--show-fps", action="store_true")
    args = ap.parse_args()

    # Choose the right type for webcam index
    source = 0 if args.source == "0" else args.source

    print(f"[info] loading model: {args.model}")
    model = YOLO(args.model)

    while True:
        frames = get_frame_generator(source, args.backend)
        got_any = False
        last_t = None
        for frame in frames:
            got_any = True
            res = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            out = res[0].plot()

            if args.show_fps:
                now = time.time()
                if last_t is None:
                    fps_txt = "FPS: —"
                else:
                    fps = 1.0 / max(1e-6, now - last_t)
                    fps_txt = f"FPS: {fps:.1f}"
                last_t = now
                cv2.putText(out, fps_txt, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("ESP32-CAM — YOLOv11 Detect", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        if got_any:
            print("[info] stream ended, reconnecting in 2s …")
            time.sleep(2)
        else:
            print("[warn] no frames received; check URL/client; retrying in 3s …")
            time.sleep(3)

if __name__ == "__main__":
    main()
