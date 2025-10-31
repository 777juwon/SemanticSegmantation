import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

def fast_hist(gt, pred, n_class, ignore_index=255):
    """
    gt, pred: HxW ndarray (uint8)
    ignore_index는 혼동 방지를 위해 미리 마스크로 제거 후 hist 계산
    """
    mask = (gt != ignore_index)
    gt = gt[mask].astype(np.int64, copy=False)
    pred = pred[mask].astype(np.int64, copy=False)
    k = gt * n_class + pred
    hist = np.bincount(k, minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_iou_from_hist(hist):
    ious = []
    for i in range(hist.shape[0]):
        tp = hist[i, i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        denom = (tp + fp + fn)
        iou = tp / denom if denom > 0 else np.nan
        ious.append(iou)
    return np.array(ious, dtype=np.float64)

def is_pred_file(p):
    # *_CategoryId.(png|jpg|jpeg) 만 평가 대상으로
    name = os.path.basename(p).lower()
    if not name.endswith(("_categoryid.png", "_categoryid.jpg", "_categoryid.jpeg")):
        return False
    return True

def load_label(path):
    # PIL로 열고 numpy uint8로
    with Image.open(path) as im:
        return np.array(im, dtype=np.uint8)

def main(args):
    label_root = os.path.abspath(args.label_dir)   # GT 라벨 루트(예: .../labelmap/val)
    pred_root  = os.path.abspath(args.pred_dir)    # 예측 라벨 루트(예: .../preds/.../label/val)
    n_class    = args.num_classes
    ignore_idx = args.ignore_index

    # 재귀적으로 예측 결과 수집
    pred_paths = sorted(
        [p for p in glob(os.path.join(pred_root, "**", "*.*"), recursive=True) if is_pred_file(p)]
    )

    if len(pred_paths) == 0:
        print(f"Found 0 prediction images under: {pred_root}")
        print("확인해주세요: prediction.py가 *_CategoryId.png(or .jpg)로 저장하는지, pred_dir가 올바른지")
        return

    # 누적 혼동행렬
    hist_total = np.zeros((n_class, n_class), dtype=np.int64)

    n_pairs = 0
    miss = 0

    for pp in tqdm(pred_paths, desc="Evaluating"):
        # pred_root 기준 상대경로를 그대로 label_root에 붙여서 GT 경로 구성
        # (예: preds/.../label/val/cam0/xxx_CategoryId.png → 학습데이터/labelmap/val/cam0/xxx_CategoryId.png)
        rel = os.path.relpath(pp, pred_root)
        gt_path = os.path.join(label_root, rel)

        # 확장자 차이 교정(png/jpg 혼용 대비)
        if not os.path.exists(gt_path):
            root, ext = os.path.splitext(gt_path)
            for cand_ext in (".png", ".jpg", ".jpeg"):
                alt = root + cand_ext
                if os.path.exists(alt):
                    gt_path = alt
                    break

        if not os.path.exists(gt_path):
            miss += 1
            continue

        gt = load_label(gt_path)
        pred = load_label(pp)

        # 예측과 GT의 크기가 다르면 GT 크기에 맞춰 리사이즈 (nearest)
        if pred.shape != gt.shape:
            with Image.open(pp) as im:
                im = im.resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST)
                pred = np.array(im, dtype=np.uint8)

        hist_total += fast_hist(gt, pred, n_class, ignore_index=ignore_idx)
        n_pairs += 1

    ious = compute_iou_from_hist(hist_total)
    miou = np.nanmean(ious) if np.isfinite(ious).any() else np.nan

    print(f"\nPairs evaluated: {n_pairs}  (missing matches: {miss})")
    print(f"📊 mIoU: {miou:.4f}" if np.isfinite(miou) else "📊 mIoU: NaN")

    for i, v in enumerate(ious):
        s = f"Class {i}: IoU = {v:.4f}" if np.isfinite(v) else f"Class {i}: IoU = NaN (ignored)"
        print(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir",  type=str, required=True,
                        help="GT label root (e.g., /workspace/학습데이터/labelmap/val)")
    parser.add_argument("--pred_dir",   type=str, required=True,
                        help="Prediction root (e.g., /workspace/preds_val_best/label/val)")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    args = parser.parse_args()
    main(args)
