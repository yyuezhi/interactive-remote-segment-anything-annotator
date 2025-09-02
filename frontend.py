#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, pickle, socket, struct
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# ============================
# Config (hard-coded switches)
# ============================
OUT_DIR = "masks_image"
os.makedirs(OUT_DIR, exist_ok=True)

SERVER_HOST = "localhost"
SERVER_PORT = 8009  # per your server

SAVE_OPTIONS = {
    "image_mask":  True,  # save image with only labeled pixels (background black)
    "overlay":     True,  # save image + transparent overlay of all labels
    "label_single":True,  # save per-label binary PNGs (one PNG per label that has pixels)
    "label_multi": True,  # save pure colored label PNG (all labels together)
}

# UI look/feel
MASK_RADIUS_DISP = 10           # brush radius in DISPLAY pixels
MASK_BLEND_ALPHA = 0.5          # overlay alpha
ZOOM_FACTOR      = 1.2          # wheel zoom step

# ============================
# Globals
# ============================
idx = 0  # current image index

# Per-image stores (parallel to id_list)
full_imgs  = []   # list of np.uint8 HxWx3 (BGR) - immutable base used for rendering
full_masks = []   # list of np.uint8 HxW label maps (0..max_labels)
points_all = []   # list: for each image -> [list() for label 0..max_labels], each item: (x_full,y_full,flag)
sizes      = []   # list of (H,W)

# Current view state (derived from full_* using viewport)
disp_img   = None  # HxWx3 BGR (resampled from full_img via viewport)
disp_mask  = None  # HxW uint8 (resampled from full_mask via viewport)

# Viewport in FULL coordinates: x0,y0,x1,y1
view_x0 = view_y0 = 0.0
view_x1 = view_y1 = 0.0

# Modes
zoom_mode_active  = False
mask_mode_active  = False
point_mode_active = False

# Zoom rectangle drag state
dragging_rect = False
rect_start = None
rect_end   = None

# Painting state
drawing_paint = False

# Labels (RGB palette; convert to BGR for cv2 when needed)
mask_colors = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 0),
]
max_labels    = len(mask_colors)
current_label = 1  # 1..max_labels

# ============================
# Utility paths/names
# ============================
def stem(p): return Path(p).stem
def path_bin_label(image_path, label): return os.path.join(OUT_DIR, f"{stem(image_path)}__L{label:02d}.png")
def path_overlay(image_path):          return os.path.join(OUT_DIR, f"{stem(image_path)}__overlay.png")
def path_image_masked(image_path):     return os.path.join(OUT_DIR, f"{stem(image_path)}__image_masked.png")
def path_labels_color(image_path):     return os.path.join(OUT_DIR, f"{stem(image_path)}__labels_color.png")


# ============================
# Socket helpers (toy protocol)
# ============================
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

def send_points_and_get_mask(full_img_bgr, points_list_for_labels, label_id, host=SERVER_HOST, port=SERVER_PORT):
    """Send (img_rgb, points[N,2], labels[N]) and get {"mask": uint8[H,W], ...}"""
    pts = points_list_for_labels[label_id]
    if not pts:
        return None
    arr = np.array(pts, dtype=np.int32)         # (N,3)
    points_np = arr[:, :2].copy().astype(np.int32)
    labels_np = arr[:, 2].copy().astype(np.int32)

    img_rgb = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB)
    payload = pickle.dumps((img_rgb, points_np, labels_np), protocol=pickle.HIGHEST_PROTOCOL)

    t0 = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        send_msg(s, payload)
        resp = recv_msg(s)
    dt = time.time() - t0

    if resp is None:
        print("[-] No response from server.")
        return None

    data = pickle.loads(resp)
    best_mask = data.get("mask", None)
    if not (isinstance(best_mask, np.ndarray) and best_mask.dtype == np.uint8):
        print("[-] Bad 'mask' in server response.")
        return None

    print(f"[server] mask_sum={int(best_mask.sum())}  time={dt:.3f}s")
    return best_mask

# ============================
# Coordinate transforms (FULL <-> DISPLAY)
# ============================
def set_view_full(image_index, center_x, center_y, zoom):
    """Set viewport by center + zoom. Keeps aspect = full W/H."""
    global view_x0, view_y0, view_x1, view_y1
    H, W = sizes[image_index]
    vw = W / zoom
    vh = H / zoom
    x0 = center_x - vw / 2
    y0 = center_y - vh / 2
    x1 = x0 + vw
    y1 = y0 + vh
    # clamp
    x0 = max(0, min(x0, W - vw))
    y0 = max(0, min(y0, H - vh))
    view_x0, view_y0, view_x1, view_y1 = x0, y0, x0 + vw, y0 + vh

def set_view_rect(image_index, x0_full, y0_full, x1_full, y1_full):
    """Set viewport to a FULL-coord rectangle, adjusted to image aspect ratio."""
    global view_x0, view_y0, view_x1, view_y1
    H, W = sizes[image_index]
    x0, x1 = sorted([float(x0_full), float(x1_full)])
    y0, y1 = sorted([float(y0_full), float(y1_full)])
    x0 = max(0.0, min(x0, W - 1)); x1 = max(x0 + 1.0, min(x1, W * 1.0))
    y0 = max(0.0, min(y0, H - 1)); y1 = max(y0 + 1.0, min(y1, H * 1.0))

    want_ar = W / H
    rw, rh = (x1 - x0), (y1 - y0)
    rect_ar = rw / rh
    if rect_ar > want_ar:
        new_h = rw / want_ar
        cy = (y0 + y1) / 2
        y0 = cy - new_h / 2
        y1 = cy + new_h / 2
    else:
        new_w = rh * want_ar
        cx = (x0 + x1) / 2
        x0 = cx - new_w / 2
        x1 = cx + new_w / 2

    x0 = max(0.0, min(x0, W - 1)); x1 = max(x0 + 1.0, min(x1, W * 1.0))
    y0 = max(0.0, min(y0, H - 1)); y1 = max(y0 + 1.0, min(y1, H * 1.0))

    view_x0, view_y0, view_x1, view_y1 = x0, y0, x1, y1

def disp_to_full(x_disp, y_disp, image_index):
    """Map display pixel -> full coords given current viewport."""
    H, W = sizes[image_index]
    x_full = view_x0 + (x_disp / W) * (view_x1 - view_x0)
    y_full = view_y0 + (y_disp / H) * (view_y1 - view_y0)
    return int(round(x_full)), int(round(y_full))

def radius_disp_to_full(r_disp, image_index):
    """Map a display-space radius to a FULL-space radius."""
    H, W = sizes[image_index]
    scale = (view_x1 - view_x0) / W  # pixels_in_full_per_disp_pixel horizontally
    return max(1, int(round(r_disp * scale)))

# ============================
# View rendering (from FULL)
# ============================
def refresh_display(image_index):
    """Rebuild display image/mask from full + viewport."""
    global disp_img, disp_mask
    H, W = sizes[image_index]
    x0i, y0i, x1i, y1i = map(lambda v: int(round(v)), (view_x0, view_y0, view_x1, view_y1))

    full = full_imgs[image_index]
    fmask = full_masks[image_index]

    crop_img  = full[y0i:y1i, x0i:x1i]
    crop_mask = fmask[y0i:y1i, x0i:x1i]

    disp_img  = cv2.resize(crop_img,  (W, H), interpolation=cv2.INTER_LINEAR)
    disp_mask = cv2.resize(crop_mask, (W, H), interpolation=cv2.INTER_NEAREST)

# ============================
# Colors & blending
# ============================
def mask_to_color(mask_u8):
    H, W = mask_u8.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    if mask_u8.max() == 0:
        return out
    colors_bgr = np.array(mask_colors, dtype=np.uint8)[:, ::-1]  # RGB->BGR
    for L in np.unique(mask_u8):
        if L == 0:
            continue
        out[mask_u8 == L] = colors_bgr[L - 1]
    return out

def blend_mask_over_image(bgr_img, lbl_mask, alpha=MASK_BLEND_ALPHA):
    overlay = mask_to_color(lbl_mask)
    return cv2.addWeighted(bgr_img, 1.0, overlay, alpha, 0)

# ============================
# Saving
# ============================
def save_image(image_index):
    """Save outputs for ONE image according to SAVE_OPTIONS."""
    img_path = id_list[image_index]
    full_bgr  = full_imgs[image_index]
    labels_u8 = full_masks[image_index]

    # 1) per-label binary PNGs
    if SAVE_OPTIONS.get("label_single", False):
        wrote = 0
        for L in range(1, max_labels + 1):
            bin_mask = (labels_u8 == L)
            if bin_mask.any():
                outp = path_bin_label(img_path, L)
                cv2.imwrite(outp, (bin_mask.astype(np.uint8) * 255))
                wrote += 1
        print(f"[save] per-label binaries: {wrote} file(s) for {img_path}")

    # 2a) overlay png (image + transparent overlay)
    if SAVE_OPTIONS.get("overlay", False):
        overlay_bgr = blend_mask_over_image(full_bgr.copy(), labels_u8, alpha=MASK_BLEND_ALPHA)
        cv2.imwrite(path_overlay(img_path), overlay_bgr)

    # 2b) image-masked png (only labeled pixels kept; background black)
    if SAVE_OPTIONS.get("image_mask", False):
        keep = (labels_u8 > 0).astype(np.uint8)[:, :, None]
        masked_bgr = (full_bgr * keep).astype(np.uint8)
        cv2.imwrite(path_image_masked(img_path), masked_bgr)

    # 3) pure label map (colorized labels only)
    if SAVE_OPTIONS.get("label_multi", False):
        labels_color = mask_to_color(labels_u8)  # BGR
        cv2.imwrite(path_labels_color(img_path), labels_color)

    print(f"[save] done for {img_path}")

def save_all_images():
    for i in range(len(id_list)):
        save_image(i)
    print("[save] ALL images saved.")

# ============================
# Mouse callback
# ============================
def on_mouse(event, x, y, flags, param):
    global dragging_rect, rect_start, rect_end, drawing_paint
    global zoom_mode_active, mask_mode_active, point_mode_active

    # ---- Zoom mode (wheel zoom + box zoom) ----
    if zoom_mode_active:
        if event == cv2.EVENT_MOUSEWHEEL:
            H, W = sizes[idx]
            cx_full, cy_full = disp_to_full(x, y, idx)
            curr_zoom = W / (view_x1 - view_x0)
            new_zoom = curr_zoom * ZOOM_FACTOR if flags > 0 else max(1.0, curr_zoom / ZOOM_FACTOR)
            set_view_full(idx, cx_full, cy_full, new_zoom)
            refresh_display(idx)

        elif event == cv2.EVENT_LBUTTONDOWN:
            dragging_rect = True
            rect_start = (x, y); rect_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and dragging_rect:
            rect_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and dragging_rect:
            dragging_rect = False
            rect_end = (x, y)
            x0f, y0f = disp_to_full(rect_start[0], rect_start[1], idx)
            x1f, y1f = disp_to_full(rect_end[0],   rect_end[1],   idx)
            set_view_rect(idx, x0f, y0f, x1f, y1f)
            refresh_display(idx)

    # ---- Paint mode ----
    elif mask_mode_active:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_paint = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing_paint = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing_paint:
            x_full, y_full = disp_to_full(x, y, idx)
            r_full = radius_disp_to_full(MASK_RADIUS_DISP, idx)
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if 1 <= current_label <= max_labels:
                    cv2.circle(full_masks[idx], (x_full, y_full), r_full, current_label, -1)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                cv2.circle(full_masks[idx], (x_full, y_full), r_full, 0, -1)
            refresh_display(idx)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            drawing_paint = False

    # ---- Point mode ----
    elif point_mode_active:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            # add point in FULL coords
            x_full, y_full = disp_to_full(x, y, idx)
            flag = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            points_all[idx][current_label].append((x_full, y_full, flag))

            # ask server for a mask and REWRITE current label
            returned_u8 = send_points_and_get_mask(full_imgs[idx], points_all[idx], current_label)
            if returned_u8 is not None:
                H, W = sizes[idx]
                assert returned_u8.shape == (H, W), f"Server mask shape {returned_u8.shape} != {(H, W)}"
                fm = full_masks[idx]
                mbool = returned_u8.astype(bool)
                # set label where mask==1
                fm[mbool] = current_label
                # clear only current-label pixels where mask==0
                fm[(~mbool) & (fm == current_label)] = 0
                refresh_display(idx)

# ============================
# Image loading and setup
# ============================
INPUT_PATH = "./input"
id_list = sorted([os.path.join(INPUT_PATH,f) for f in os.listdir(INPUT_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
if not id_list:
    raise RuntimeError("No images found in current directory (png/jpg/jpeg).")

# Preload all images (keep sizes as-is; window will match each image's real size)
SCALE_X = 2.0  # example: expand image 2x in both width and height

for p in id_list:
    arr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]  # to BGR

    if SCALE_X != 1.0:
        new_W = int(arr.shape[1] * SCALE_X)
        new_H = int(arr.shape[0] * SCALE_X)
        arr = cv2.resize(arr, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    H, W = arr.shape[:2]
    full_imgs.append(arr.copy())
    full_masks.append(np.zeros((H, W), dtype=np.uint8))
    points_all.append([[] for _ in range(max_labels + 1)])
    sizes.append((H, W))

def init_view_to_full(image_index):
    H, W = sizes[image_index]
    set_view_full(image_index, W / 2, H / 2, zoom=1.0)
    refresh_display(image_index)

init_view_to_full(idx)

# ============================
# UI loop
# ============================
cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
# Start the window at the REAL image size (no forced 1000x1000)
H0, W0 = sizes[idx]
cv2.resizeWindow("annotator", W0, H0)
cv2.setMouseCallback("annotator", on_mouse)

print("Keys:")
print("  w = previous image")
print("  s = save current image, then next image")
print("  a = label - (clamped, no wrap)")
print("  d = label + (clamped, no wrap)")
print("  z = zoom mode (mouse wheel or box-zoom drag)")
print("  x = point mode (LMB=positive, RMB=negative)")
print("  q = paint mode (LMB=paint, RMB=erase)")
print("  e = clear current label for current image")
print("  c = reset current image to FULL view and CLEAR ALL labels/points")
print("  r = save ALL images (based on SAVE_OPTIONS)")
print("  ESC = exit")

while True:
    # prepare frame
    frame = disp_img.copy()

    # draw current label points within view
    if points_all[idx][current_label]:
        H, W = sizes[idx]
        vw = (view_x1 - view_x0)
        vh = (view_y1 - view_y0)
        for (xf, yf, flag) in points_all[idx][current_label]:
            if (view_x0 <= xf < view_x1) and (view_y0 <= yf < view_y1):
                xd = int(round((xf - view_x0) / vw * W))
                yd = int(round((yf - view_y0) / vh * H))
                cv2.circle(frame, (xd, yd), 4, (255, 0, 0) if flag == 1 else (0, 0, 255), -1)

    # draw rectangle during drag
    if zoom_mode_active and dragging_rect and rect_start and rect_end:
        cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)

    # overlay transparent mask
    frame = blend_mask_over_image(frame, disp_mask, alpha=MASK_BLEND_ALPHA)
    cv2.imshow("annotator", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        print("Exiting…")
        break

    elif key == ord('w'):
        if idx - 1 >= 0:
            idx -= 1
            init_view_to_full(idx)
            Hn, Wn = sizes[idx]
            cv2.resizeWindow("annotator", Wn, Hn)
            print(f"[prev] {id_list[idx]}")
        else:
            print("Already first.")

    elif key == ord('s'):
        # SAVE current image, then NEXT
        save_image(idx)
        if idx + 1 < len(id_list):
            idx += 1
            init_view_to_full(idx)
            Hn, Wn = sizes[idx]
            cv2.resizeWindow("annotator", Wn, Hn)
            print(f"[next] {id_list[idx]}")
        else:
            print("Already last.")

    elif key == ord('a'):
        if current_label > 1:
            current_label -= 1
        print(f"[label] {current_label}")

    elif key == ord('d'):
        if current_label < max_labels:
            current_label += 1
        print(f"[label] {current_label}")

    elif key == ord('z'):
        zoom_mode_active  = True
        mask_mode_active  = False
        point_mode_active = False
        print("[mode] zoom")

    elif key == ord('x'):
        point_mode_active = True
        mask_mode_active  = False
        zoom_mode_active  = False
        print("[mode] point (LMB=+, RMB=-)")

    elif key == ord('q'):
        mask_mode_active  = True
        zoom_mode_active  = False
        point_mode_active = False
        print("[mode] paint (LMB=paint, RMB=erase)")

    elif key == ord('e'):
        # clear CURRENT LABEL for current image
        L = current_label
        full_masks[idx][full_masks[idx] == L] = 0
        points_all[idx][L].clear()
        refresh_display(idx)
        print(f"[clear] label {L} cleared for {id_list[idx]}")

    elif key == ord('c'):
        # reset current image to FULL view AND CLEAR ALL labels/points
        full_masks[idx][:] = 0
        for L in range(0, max_labels + 1):
            points_all[idx][L].clear()
        Hn, Wn = sizes[idx]
        set_view_full(idx, Wn / 2, Hn / 2, zoom=1.0)
        refresh_display(idx)
        print(f"[reset] cleared all labels & points, full view for {id_list[idx]}")

    elif key == ord('r'):
        # SAVE ALL images according to SAVE_OPTIONS
        save_all_images()

cv2.destroyAllWindows()
