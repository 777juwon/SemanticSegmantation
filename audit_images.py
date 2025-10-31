# audit_images.py
import os, cv2, csv, glob, numpy as np
from pathlib import Path

IMG_DIR = "/workspace/학습데이터/image/train"   # 네 경로로 변경 가능
OUT_CSV = "/workspace/runs/image_audit.csv"

def metrics(img):
    # img: BGR uint8
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(np.float32)/255.0
    s = hsv[:,:,1].astype(np.float32)/255.0

    # 밝기/채도/대비
    bright = float(v.mean())
    contrast = float(v.std())
    saturation = float(s.mean())

    # 블러 지표(라플라시안 분산; 작을수록 흐림)
    blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    # 안개(헤이즈) 지표: Dark Channel Prior의 평균(클수록 헤이즈 많다고 볼 수 있음)
    dark = cv2.min(cv2.min(img[:,:,0], img[:,:,1]), img[:,:,2]).astype(np.float32)/255.0
    haze_index = float(dark.mean())

    # JPEG 블록/고주파 에너지로 아주 러프하게 추정(값이 낮으면 하이프리퀀시 적음=블러/헤이즈 가능)
    high = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1, ksize=3)
    hf_energy = float(np.mean(np.abs(high)))

    return bright, contrast, saturation, blur, haze_index, hf_energy

paths = []
for ext in ("*.jpg", "*.png", "*.jpeg"):
    paths += glob.glob(os.path.join(IMG_DIR, "**", ext), recursive=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path","bright","contrast","saturation","blur_var","haze_idx","hf_energy"])
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: continue
        b, c, s, bl, hz, hf = metrics(img)
        w.writerow([p, b, c, s, bl, hz, hf])

print(f"[OK] Wrote {OUT_CSV} with {len(paths)} rows")
