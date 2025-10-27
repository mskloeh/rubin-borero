






# app.py — Streamlit CFOP coach WITH Twisty (cubing.js) embedded on the left
from __future__ import annotations


import json, time, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


import requests
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components


# ---------------- Page ----------------
st.set_page_config(page_title="Rubik's Cube AI — ESP32-CAM + CFOP", layout="wide")
st.title("CFOP coach")
st.caption(
    "Left: cubing.js Twisty Player. Right: scan faces (U,R,F,D,L,B) → label → 3D viewer → CFOP helper. "
    "Live MJPEG supported via /stream."
)


# ---------------- Constants / Data ----------------
FACES_ORDER = ["U", "R", "F", "D", "L", "B"]  # URFDLB order
FACE_TO_COLOR = {
    "U": "#ffffff",
    "R": "#ff0000",
    "F": "#00ff00",
    "D": "#ffff00",
    "L": "#ffa500",
    "B": "#0000ff",
}


@dataclass
class FaceScan:
    face: str
    image_bgr: np.ndarray
    centers_lab: Dict[str, np.ndarray]      # {face: LAB centroid of center sticker}
    stickers_lab: List[np.ndarray]          # 9 LAB samples
    assigned_labels: List[str]              # 9 labels inferred from centroids


if "scans" not in st.session_state:
    st.session_state.scans: Dict[str, FaceScan] = {}
if "twisty_alg" not in st.session_state:
    st.session_state.twisty_alg = "R U R' U R U2' R'"


# ---------------- Helpers: image/color ----------------
def _pil_to_bgr(pil_img) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def _sample_3x3_stickers(img_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    side = int(min(h, w) * 0.8)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    cell = side // 3
    stickers_lab: List[np.ndarray] = []
    img_lab = _bgr_to_lab(img_bgr)
    for r in range(3):
        for c in range(3):
            cx = x0 + c * cell + cell // 2
            cy = y0 + r * cell + cell // 2
            half = max(6, cell // 10)
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            patch = img_lab[y1:y2, x1:x2]
            mean_lab = patch.reshape(-1, 3).mean(axis=0)
            stickers_lab.append(mean_lab)
    return stickers_lab


def _overlay_grid(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.copy()
    h, w = img.shape[:2]
    side = int(min(h, w) * 0.8)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    cell = side // 3
    cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (40, 200, 255), 2)
    for i in range(1, 3):
        cv2.line(img, (x0, y0 + i * cell), (x0 + side, y0 + i * cell), (40, 200, 255), 2)
        cv2.line(img, (x0 + i * cell, y0), (x0 + i * cell, y0 + side), (40, 200, 255), 2)
    return img


# ---------------- ESP32-CAM ----------------
def grab_from_esp32(base_url: str) -> Optional[np.ndarray]:
    url = base_url.rstrip("/") + "/capture"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        buf = np.frombuffer(r.content, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return bgr
    except Exception as e:
        st.error(f"Failed to fetch from {url}: {e}")
        return None


def show_mjpeg(base_url: str, width=640, height=480) -> None:
    src = base_url.rstrip("/") + "/stream"
    html = f'<img src="{src}" width="{width}" height="{height}" />'
    components.html(html, height=height + 10)


def pseudo_live(base_url: str, seconds=8, interval=0.25) -> None:
    placeholder = st.empty()
    t0 = time.time()
    while time.time() - t0 < seconds:
        bgr = grab_from_esp32(base_url)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, caption="ESP32-CAM (pseudo-live)")
        time.sleep(interval)


# ---------------- Labeling ----------------
def _lab_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def assign_labels(stickers_lab: List[np.ndarray], centroids: Dict[str, np.ndarray]) -> List[str]:
    labels: List[str] = []
    for s in stickers_lab:
        best_face = None
        best_d = 1e9
        for face, ctr in centroids.items():
            d = _lab_dist(s, ctr)
            if d < best_d:
                best_d = d
                best_face = face
        labels.append(best_face or "U")
    return labels


def build_state_string(scans: Dict[str, FaceScan]) -> str:
    out: List[str] = []
    for face in FACES_ORDER:
        s = scans[face].assigned_labels if face in scans else [face] * 9
        out.extend(s)
    return "".join(out)


# ---------------- Twisty (cubing.js) inside Streamlit ----------------
def render_twisty(alg: str) -> None:
    """Renders your cubing.js Twisty Player snippet inside Streamlit."""
    html = f"""
<div style="padding:6px;border:1px solid #263041;border-radius:12px;background:#0b1220">
  <script type="module" src="https://cdn.cubing.net/v0/js/cubing/twisty"></script>
  <twisty-player id="twisty"
    puzzle="3x3x3"
    background="none"
    camera-distance="3.5"
    fov="38"
    control-panel="experimental"
    alg="{alg}">
  </twisty-player>
  <div style="color:#94a3b8;font:13px/1.4 system-ui;margin-top:8px">
    This is an example for the
    <a href="https://js.cubing.net/cubing/#show-alg" target="_blank" style="color:#93c5fd">cubing.js documentation</a>.
  </div>
</div>
"""
    components.html(html, height=420)


# ---------------- 3D viewer (Three.js via components.html) ----------------
def render_threejs(state_str: str) -> None:
    data_js = json.dumps(list(state_str))
    palette_js = json.dumps(FACE_TO_COLOR)
    html = """
<div id="cube-root"></div>
<script src="https://unpkg.com/three@0.157.0/build/three.min.js"></script>
<script>
const state = %%STATE%%;
const palette = %%PALETTE%%;


const W = 500, H = 420;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, W/H, 0.1, 1000);
camera.position.set(4.5, 5.5, 6.5);


const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W, H);
const root = document.getElementById('cube-root');
root.innerHTML = '';
root.appendChild(renderer.domElement);


const light = new THREE.DirectionalLight(0xffffff, 1.0); light.position.set(5,7,9);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));


const gap = 0.02;
const cubelet = new THREE.BoxGeometry(1-gap,1-gap,1-gap);


function addSticker(mesh, face, color){
  const s = 0.30; const off = 0.51;
  const stickerGeo = new THREE.PlaneGeometry(s, s);
  const mat = new THREE.MeshBasicMaterial({color: color, side: THREE.DoubleSide});
  const st = new THREE.Mesh(stickerGeo, mat);
  if(face==='U'){ st.position.y = off; st.rotation.x = -Math.PI/2; }
  if(face==='D'){ st.position.y = -off; st.rotation.x =  Math.PI/2; }
  if(face==='F'){ st.position.z = off; }
  if(face==='B'){ st.position.z = -off; st.rotation.y = Math.PI; }
  if(face==='R'){ st.position.x = off; st.rotation.y = -Math.PI/2; }
  if(face==='L'){ st.position.x = -off; st.rotation.y =  Math.PI/2; }
  mesh.add(st);
}


const faces = {};
let idx=0; ['U','R','F','D','L','B'].forEach(f=>{ faces[f]=state.slice(idx, idx+9); idx+=9; });


for(let y=1;y>=-1;y--){
  for(let z=1;z>=-1;z--){
    for(let x=-1;x<=1;x++){
      if(x===0 && y===0 && z===0) continue;
      const m = new THREE.Mesh(cubelet, new THREE.MeshStandardMaterial({color:0x111111}));
      m.position.set(x, y, z);


      if (y===1) { const r = 0; const c = x+1; const i = r*3 + c; addSticker(m,'U',palette[faces['U'][i]]); }
      if (y===-1){ const r = 2; const c = x+1; const i = r*3 + c; addSticker(m,'D',palette[faces['D'][i]]); }
      if (z===1) { const r = 2 - (y+1); const c = x+1; const i = r*3 + c; addSticker(m,'F',palette[faces['F'][i]]); }
      if (z===-1){ const r = 2 - (y+1); const c = 2 - (x+1); const i = r*3 + c; addSticker(m,'B',palette[faces['B'][i]]); }
      if (x===1) { const r = 2 - (y+1); const c = 2 - (z+1); const i = r*3 + c; addSticker(m,'R',palette[faces['R'][i]]); }
      if (x===-1){ const r = 2 - (y+1); const c = z+1; const i = r*3 + c; addSticker(m,'L',palette[faces['L'][i]]); }


      scene.add(m);
    }
  }
}


function animate(){
  requestAnimationFrame(animate);
  scene.rotation.y += 0.003;
  scene.rotation.x += 0.0015;
  renderer.render(scene, camera);
}
animate();
</script>
"""
    html = html.replace("%%STATE%%", data_js).replace("%%PALETTE%%", palette_js)
    components.html(html, height=440)


# ---------------- CFOP helper (very simple: 2-Look OLL edges) ----------------
def analyze_last_layer_simple(state_str: str) -> Tuple[str, str]:
    U = state_str[:9]
    edges = [U[1], U[3], U[5], U[7]]
    n_up = sum(1 for e in edges if e == 'U')
    if n_up == 2:
        if (U[1] == 'U' and U[7] == 'U') or (U[3] == 'U' and U[5] == 'U'):
            return ("Line", "Hold line horizontal, then:  F R U R' U' F'")
        else:
            return ("L-shape", "Place L at back-left (UL & UB), then:  F U R U' R' F'")
    elif n_up == 0:
        return ("Dot", "Do:  F R U R' U' F'  then repeat for Line or L.")
    else:
        return ("Mixed", "Edges partly oriented; use Line/L algs to finish.")


# ---------------- Layout ----------------
col_left, col_right = st.columns([1.05, 1])


# LEFT: Twisty viewer and controls
with col_left:
    st.subheader("Algorithm Viewer (cubing.js)")
    presets = {
        "": "",
        "Sune": "R U R' U R U2' R'",
        "Antisune": "R U2 R' U' R U' R'",
        "T-Perm": "R U R' U' R' F R2 U' R' U' R U R' F'",
        "OLL Line": "F R U R' U' F'",
        "OLL L-shape": "F U R U' R' F'",
    }
    c1, c2 = st.columns([3, 2])
    alg_in = c1.text_input("Singmaster alg", st.session_state.twisty_alg, key="twisty_alg_input")
    preset = c2.selectbox("Presets", list(presets.keys()), index=0)
    if preset and presets[preset]:
        st.session_state.twisty_alg = presets[preset]
        st.rerun()
    apply = st.button("Apply to Twisty")
    if apply:
        st.session_state.twisty_alg = alg_in
        st.rerun()


    render_twisty(st.session_state.twisty_alg)


# RIGHT: capture → state → 3D → CFOP
with col_right:
    st.subheader("Scan, 3D view, CFOP helper")
    st.sidebar.header("Capture source")
    source = st.sidebar.radio("Pick source:", ["ESP32-CAM (HTTP)", "Local webcam"], index=0)
    esp_url = st.sidebar.text_input("ESP32-CAM base URL", "http://esp32cam.local")
    st.sidebar.caption("PC and ESP32-CAM must be on the same Wi-Fi. Sketch exposes /capture and /stream.")
    st.sidebar.header("Scan order")
    st.sidebar.write("Scan faces in this order: **U, R, F, D, L, B**. Center sticker defines that face's color.")


    face_choice = st.selectbox("Face to capture", FACES_ORDER, index=0, key="face_select")


    if source == "ESP32-CAM (HTTP)":
        live = st.checkbox("Live preview (MJPEG via /stream)")
        if live:
            show_mjpeg(esp_url, width=640, height=480)


    bgr = None
    if source == "ESP32-CAM (HTTP)":
        c1, c2 = st.columns(2)
        if c1.button("📸 Capture from ESP32-CAM"):
            bgr = grab_from_esp32(esp_url)
        if c2.button("▶ Pseudo-live (8s)"):
            pseudo_live(esp_url, seconds=8, interval=0.25)
    else:
        pic = st.camera_input(f"Take a clear, front-on photo of the {face_choice} face")
        if pic is not None:
            bgr = _pil_to_bgr(pic)


    if bgr is not None:
        grid = _overlay_grid(bgr)
        st.image(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), caption=f"{face_choice} with 3×3 grid")


        stickers_lab = _sample_3x3_stickers(bgr)
        center_lab = stickers_lab[4]


        # Existing centroids + this face
        centroids: Dict[str, np.ndarray] = {}
        for f in FACES_ORDER:
            if f in st.session_state.scans:
                centroids[f] = st.session_state.scans[f].centers_lab[f]
        centroids[face_choice] = center_lab


        assigned = assign_labels(stickers_lab, centroids or {face_choice: center_lab})
        st.write("Assigned labels (preview):", " ".join(assigned))


        st.session_state.scans[face_choice] = FaceScan(
            face=face_choice,
            image_bgr=bgr,
            centers_lab={face_choice: center_lab},
            stickers_lab=stickers_lab,
            assigned_labels=assigned,
        )


    have_all = all(f in st.session_state.scans for f in FACES_ORDER)
    if have_all:
        # Recompute labels with full centroids
        full_centroids = {f: st.session_state.scans[f].centers_lab[f] for f in FACES_ORDER}
        for f in FACES_ORDER:
            fs = st.session_state.scans[f]
            st.session_state.scans[f].assigned_labels = assign_labels(fs.stickers_lab, full_centroids)


        state = build_state_string(st.session_state.scans)
        st.text_area("URFDLB state (54 chars)", state, height=70)
        render_threejs(state)


        st.markdown("---")
        st.subheader("2-Look OLL — quick read")
        pat, tip = analyze_last_layer_simple(state)
        st.write(f"Edge pattern: **{pat}**")
        st.code(tip)


        # Extract alg from tip and let user send it to Twisty
        m = re.search(r":\\s*([^:]+)$", tip)
        alg = m.group(1).strip() if m else ""
        if alg:
            st.text_input("Suggested alg", alg, key="suggested_alg", disabled=True)
            if st.button("➡️ Send to Twisty (left)"):
                st.session_state.twisty_alg = alg
                st.rerun()
        else:
            st.caption("No discrete alg parsed from this tip.")
    else:
        missing = [f for f in FACES_ORDER if f not in st.session_state.scans]
        st.warning("Waiting for faces: " + ", ".join(missing))


st.markdown(
    """
---
**Tips**
- If live preview stutters, set camera FRAMESIZE_QVGA and jpeg_quality 12–15.
- If /stream or /capture time out on public Wi-Fi, test on a phone hotspot or home router (cafés often enable client isolation).
- You can also use http://esp32cam.local if mDNS is printed in Serial.
"""
)



