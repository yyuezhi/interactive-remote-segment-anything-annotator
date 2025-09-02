import os, socket, struct, threading, pickle
from datetime import datetime
import numpy as np
from PIL import Image
import cv2  # for resize, contours, drawing

# ── SAM2 ───────────────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── config ─────────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8003
UPLOAD_DIR = "received_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG        = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_DEVICE      = "cuda"   # or "cpu"

# ── socket helpers (length-prefixed) ───────────────────────────────────
def send_msg(conn, data_bytes):
    conn.sendall(struct.pack(">Q", len(data_bytes)) + data_bytes)

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def recv_msg(conn):
    header = recv_exact(conn, 8)
    if header is None:
        return None
    (length,) = struct.unpack(">Q", header)
    if length == 0:
        return b""
    return recv_exact(conn, length)

# ── SAM2 model (load once) ─────────────────────────────────────────────
print("[*] Loading SAM2…")
_sam2_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT, device=SAM_DEVICE)
_predictor = SAM2ImagePredictor(_sam2_model)
print("[+] SAM2 ready.")

# ── utilities: visualization and scoring ───────────────────────────────
def _save_visual(image_rgb, mask_u8, points_np, labels_np, borders=True, alpha=0.6):
    """
    Overlay a transparent mask + draw white border + draw points (green=pos, red=neg).
    Saves to UPLOAD_DIR/vis_<timestamp>.png
    """
    H, W = image_rgb.shape[:2]
    # Safety: ensure shapes match
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # Work in BGR for OpenCV drawing
    vis_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = mask_u8.astype(np.uint8)
    mask_bool = mask_u8.astype(bool)

    # Transparent overlay color (same as toy example: RGB [30,144,255]) -> BGR [255,144,30]
    color_bgr = np.array([255, 144, 30], dtype=np.uint8)
    color_img = np.zeros_like(vis_bgr, dtype=np.uint8)
    color_img[:] = color_bgr

    # Blend only on masked pixels
    blended = cv2.addWeighted(vis_bgr, 1.0, color_img, alpha, 0)
    vis_bgr[mask_bool] = blended[mask_bool]

    # Optional white border
    if borders:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        cv2.drawContours(vis_bgr, contours, -1, (255, 255, 255), thickness=2)

    # Points: green=positive(1), red=negative(0)
    if isinstance(points_np, np.ndarray) and points_np.size > 0:
        pts = points_np.astype(int)
        labs = labels_np.astype(int)
        for (x, y), lab in zip(pts, labs):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(vis_bgr, (int(x), int(y)), 5, color, -1)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(UPLOAD_DIR, f"vis_{ts}.png")
    cv2.imwrite(out_path, vis_bgr)
    print(f"[+] Saved visualization: {out_path}")

def _select_best_idx_by_logits(masks_uint8, logits_f32):
    """
    masks_uint8: [K,Hm,Wm] in {0,1}, image resolution
    logits_f32 : [K,Hl,Wl], may be different resolution than masks
    Strategy: resize each logits[k] to mask[k] size and compute masked-mean.
    """
    K, Hm, Wm = masks_uint8.shape
    scores = np.full(K, -np.inf, dtype=np.float64)
    for k in range(K):
        mask_k = masks_uint8[k].astype(bool)       # (Hm,Wm)
        logit_k = logits_f32[k]                    # (Hl,Wl)
        if logit_k.shape != mask_k.shape:
            logit_k = cv2.resize(logit_k, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
        if mask_k.any():
            scores[k] = float(logit_k[mask_k].mean())
    return int(np.argmax(scores))

# ── request handler ────────────────────────────────────────────────────
def handle_client(conn, addr):
    try:
        data = recv_msg(conn)
        if data is None:
            print(f"[-] {addr} closed without data.")
            return

        # Expect (image_np, points_np, labels_np)
        try:
            image_np, points_np, labels_np = pickle.loads(data)
        except Exception as e:
            print(f"[-] Unpickle failed: {e}")
            return

        # Validate
        if not isinstance(image_np, np.ndarray) or image_np.ndim not in (2, 3):
            print("[-] Invalid image array.")
            return
        if not isinstance(points_np, np.ndarray) or points_np.ndim != 2 or points_np.shape[1] != 2:
            print("[-] points must be shape [N,2].")
            return
        if not isinstance(labels_np, np.ndarray) or labels_np.ndim != 1 or labels_np.shape[0] != points_np.shape[0]:
            print("[-] labels must be shape [N].")
            return

        # Ensure RGB uint8
        if image_np.ndim == 2:
            image_rgb = np.stack([image_np]*3, axis=-1)
        else:
            if image_np.shape[2] >= 3:
                image_rgb = image_np[..., :3]
            else:
                pad = np.zeros((*image_np.shape[:2], 3), dtype=image_np.dtype)
                pad[..., :image_np.shape[2]] = image_np
                image_rgb = pad
        image_rgb = image_rgb.astype(np.uint8)

        # Run SAM2 using client-provided clicks
        _predictor.set_image(image_rgb)
        masks, scores, logits = _predictor.predict(
            point_coords=points_np.astype(np.int32),
            point_labels=labels_np.astype(np.int32),
            multimask_output=True,
        )
        masks  = masks.astype(np.uint8)     # [K,H,W] {0,1} at image res
        logits = logits.astype(np.float32)  # [K,h,w] (often lower res)

        best_idx  = _select_best_idx_by_logits(masks, logits)
        best_mask = masks[best_idx]         # [H,W] uint8

        # >>> NEW: save visualization (image + mask + points)
        _save_visual(image_rgb, best_mask, points_np, labels_np, borders=True, alpha=0.6)

        # Return best mask and echo the points/labels used
        payload = {"mask": best_mask, "points": points_np, "labels": labels_np}
        resp = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        send_msg(conn, resp)
        print(f"[+] Sent best mask to {addr} | mask={best_mask.shape}  clicks={points_np.shape[0]}")

    except Exception as e:
        print(f"[-] Error handling {addr}: {e}")
    finally:
        conn.close()

# ── server loop ────────────────────────────────────────────────────────
def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f"[*] Listening on {host}:{port}")
    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr))
            t.daemon = True
            t.start()
    except KeyboardInterrupt:
        print("\n[!] Shutting down.")
    finally:
        srv.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default=SAM_DEVICE)
    args = ap.parse_args()
    start_server(args.host, args.port)