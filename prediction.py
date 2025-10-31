# prediction.py
import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from DDRNet import DDRNet


# ====== 컬러맵 (시각화용) ======
COLORMAP = np.array([
    [ 64,  64,  64], [192, 192, 192], [255, 255,   0], [255, 200,   0],
    [128,  64,  64], [ 64, 128,  64], [  0,  64, 255], [  0,  32, 192],
    [  0,   0, 160], [128,   0, 128], [ 64,   0,  64], [255,   0,   0],
    [192,   0,   0], [255, 128,   0], [  0, 255, 255], [  0, 192, 192],
    [  0, 255,   0], [  0, 192,   0], [255,   0, 255],
], dtype=np.uint8)


# ========================== Dataset ==========================
class TestSegmentationDataset(Dataset):
    """
    dataset_dir/image/<split>/cam*/<file>.* 구조를 읽어서
    학습과 동일한 Normalize(mean/std)까지 적용합니다.
    """
    def __init__(self, root_dir, split='val', input_size=None, keep_ratio=False,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.image_dir = os.path.join(root_dir, "image", split)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*.*")))
        self.input_size = tuple(input_size) if input_size is not None else None
        self.keep_ratio = keep_ratio

        # 학습과 동일: ToTensor + Normalize
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.image_paths)

    def _resize_or_letterbox(self, img, size_hw):
        """ size_hw=(H,W). keep_ratio=True면 letterbox 패딩으로 종횡비 유지. """
        H, W = size_hw
        if not self.keep_ratio:
            return img.resize((W, H), resample=Image.Resampling.BILINEAR)

        # letterbox
        ow, oh = img.size
        scale = min(W / ow, H / oh)
        nw, nh = int(round(ow * scale)), int(round(oh * scale))
        img_rs = img.resize((nw, nh), resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        left = (W - nw) // 2
        top = (H - nh) // 2
        canvas.paste(img_rs, (left, top))
        return canvas

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        orig_w, orig_h = img.size

        if self.input_size is not None:
            img = self._resize_or_letterbox(img, self.input_size)

        # ToTensor + Normalize (학습과 일치)
        t = transforms.functional.to_tensor(img)
        t = self.normalize(t)

        return t, p, (orig_h, orig_w)


# ========================== Model load ==========================
def load_model(weight_path, num_classes, device):
    model = DDRNet(num_classes=num_classes)
    model = torch.nn.DataParallel(model).to(device)

    state = torch.load(weight_path, map_location=device)
    has_module = any(k.startswith("module.") for k in state.keys())
    needs_module = any(k.startswith("module.") for k in model.state_dict().keys())
    if has_module != needs_module:
        if needs_module:
            state = {"module." + k: v for k, v in state.items()}
        else:
            state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[load_model] missing keys (head-up):", list(missing)[:5])
    if unexpected:
        print("[load_model] unexpected keys (head-up):", list(unexpected)[:5])

    model.eval()
    return model


# ========================== Save helpers ==========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_label_png(pred_np, save_path_png):
    ensure_dir(os.path.dirname(save_path_png))
    Image.fromarray(pred_np).save(save_path_png)

def save_colormap_jpg(pred_np, save_path_jpg):
    color = COLORMAP[pred_np]
    ensure_dir(os.path.dirname(save_path_jpg))
    Image.fromarray(color).save(save_path_jpg, quality=95)


# ========================== Inference ==========================
def infer_one(model, img_tensor, device, out_hw=None):
    with torch.no_grad():
        logits = model(img_tensor.to(device, non_blocking=True))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if out_hw is not None:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1)  # [1,H,W]
        return pred.squeeze(0).cpu().numpy().astype(np.uint8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    ds = TestSegmentationDataset(
        root_dir=args.dataset_dir,
        split=args.split,
        input_size=args.input_size,
        keep_ratio=args.keep_ratio,
        mean=tuple(args.mean), std=tuple(args.std),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = load_model(args.weight_path, args.num_classes, device)

    label_root = os.path.join(args.result_dir, "label", args.split)
    color_root = os.path.join(args.result_dir, "colormap", args.split)

    for img_tensor, img_path, (orig_h, orig_w) in tqdm(dl, desc="Predicting..."):
        img_path = img_path[0]
        pred_np = infer_one(model, img_tensor, device, out_hw=(int(orig_h), int(orig_w)))

        rel = os.path.relpath(img_path, os.path.join(args.dataset_dir, "image", args.split))
        cam = rel.split(os.sep)[0]
        stem, _ = os.path.splitext(os.path.basename(rel))

        save_png = os.path.join(label_root, cam, f"{stem}_CategoryId.png")
        save_jpg = os.path.join(color_root, cam, f"{stem}.jpg")
        save_label_png(pred_np, save_png)
        save_colormap_jpg(pred_np, save_jpg)

    print(f"[Done] Labels -> {label_root}")
    print(f"[Done] Colormaps -> {color_root}")


# ========================== CLI ==========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", help="image/<split>/... (예: val, test)")
    ap.add_argument("--weight_path", type=str, required=True)
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--input_size", type=int, nargs=2, default=[1024, 1024],
                    help="모델 입력 크기 (H W). 원본 크기로 복원 저장됨")
    ap.add_argument("--keep_ratio", action="store_true",
                    help="리사이즈 시 종횡비 유지(letterbox 패딩)")
    # 학습 파이프라인과 100% 일치하도록 mean/std를 명시 가능 (기본은 ImageNet)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std",  type=float, nargs=3, default=[0.229, 0.224, 0.225])
    args = ap.parse_args()
    main(args)

