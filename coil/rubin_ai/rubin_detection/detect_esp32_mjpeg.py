# detect_esp32_mjpeg.py
import requests, numpy as np, cv2, time
from ultralytics import YOLO

URL = "http://192.168.209.176/stream"  # <-- your working browser URL
MODEL = r"C:\coil\Druns\rubin_y11s_960_e150\weights\best.pt"


model = YOLO(MODEL)
buf = b""
fps_t = None

with requests.get(URL, stream=True, timeout=10) as r:
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=2048):
        if not chunk:
            continue
        buf += chunk
        a = buf.find(b'\xff\xd8')  # SOI
        b = buf.find(b'\xff\xd9')  # EOI
        if a != -1 and b != -1 and b > a:
            jpg = buf[a:b+2]
            buf = buf[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # YOLO inference (tune imgsz/conf for ESP32-CAM)
            res = model(frame, imgsz=960, conf=0.25, verbose=False)
            out = res[0].plot()

            now = time.time()
            if fps_t is None: fps_text = "FPS: —"
            else: fps_text = f"FPS: {1.0/(now - fps_t):.1f}"
            fps_t = now
            cv2.putText(out, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("ESP32-CAM — YOLO", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
