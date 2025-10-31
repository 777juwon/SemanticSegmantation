# train.py
import os
import math
import argparse
from typing import Optional, Tuple, List
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Pillow Resampling 호환 (구/신버전)
try:
    RESIZE_BILINEAR = Image.Resampling.BILINEAR
    RESIZE_NEAREST  = Image.Resampling.NEAREST
except AttributeError:
    RESIZE_BILINEAR = Image.BILINEAR
    RESIZE_NEAREST  = Image.NEAREST

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

from DDRNet import DDRNet


# ===================== Constants / Utils =====================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 19
IGNORE_INDEX  = 255

COLORMAP = np.array([
    ( 64,  64,  64), (192, 192, 192), (255, 255,   0), (255, 200,   0),
    (128,  64,  64), ( 64, 128,  64), (  0,  64, 255), (  0,  32, 192),
    (  0,   0, 160), (128,   0, 128), ( 64,   0,  64), (255,   0,   0),
    (192,   0,   0), (255, 128,   0), (  0, 255, 255), (  0, 192, 192),
    (  0, 255,   0), (  0, 192,   0), (255,   0, 255)
], dtype=np.uint8)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inv_normalize(img_t: torch.Tensor) -> torch.Tensor:
    """[3,H,W] -> [3,H,W] uint8"""
    mean = torch.tensor(IMAGENET_MEAN, device=img_t.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=img_t.device)[:, None, None]
    x = img_t * std + mean
    return (x.clamp(0, 1) * 255).byte().cpu()

def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    mask_np: (H,W) uint8 (0..18 & 255)
    - 0..18 은 팔레트, 255는 회색(무시)
    """
    out = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    valid = (mask_np != IGNORE_INDEX) & (mask_np < NUM_CLASSES)
    out[valid] = COLORMAP[mask_np[valid]]
    out[mask_np == IGNORE_INDEX] = (128, 128, 128)
    return out


# ===================== Dataset =====================
class SegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        crop_hw: Tuple[int, int],
        scale_range: Tuple[float, float],
        args,                           # argparse 전체 전달 (photometric 설정 등)
        verbose: bool = True
    ):
        self.args = args
        self.root = Path(root_dir)
        self.split = split
        self.crop_hw = crop_hw
        self.scale_range = scale_range

        self.img_dir = self.root / "image" / split
        self.lab_dir = self.root / "labelmap" / split

        # 모든 cam*/ 하위 이미지
        img_paths = sorted(p for p in self.img_dir.glob("cam*/*.*") if p.is_file())

        self.pairs: List[Tuple[Path, Path]] = []
        missing = 0
        for ip in img_paths:
            lp = self._find_label_for_image(ip)
            if lp is not None and lp.exists():
                self.pairs.append((ip, lp))
            else:
                missing += 1

        if verbose:
            print(f"[Dataset] split={split} images={len(img_paths)} -> valid_pairs={len(self.pairs)} missing={missing}")

        self.mean = IMAGENET_MEAN
        self.std  = IMAGENET_STD

        # strong_list를 photometric/샘플러 둘 다에서 쓰기 위해 set으로 보관
        self.strong_keys = getattr(args, "_strong_keys", [])

        # 희귀 클래스 파라미터
        self.rare_ids    = args.rare_ids
        self.crop_retry  = args.crop_retry
        self.min_rare_px = args.min_rare_px

    def _find_label_for_image(self, img_path: Path) -> Optional[Path]:
        """여러 네이밍 케이스 지원."""
        rel = img_path.relative_to(self.img_dir)  # camX/xxx.ext
        subdir = rel.parent
        stem = img_path.stem  # 파일명 without ext

        candidates = [f"{stem}_CategoryId.png"]
        if stem.endswith("_leftImg8bit"):
            no_left = stem.replace("_leftImg8bit", "")
            gt_fine = stem.replace("_leftImg8bit", "_gtFine")
            candidates.extend([f"{no_left}_CategoryId.png", f"{gt_fine}_CategoryId.png"])

        for name in candidates:
            lp = self.lab_dir / subdir / name
            if lp.exists():
                return lp
        return None

    # ---------- geometric aug ----------
    def _random_scale(self, img: Image.Image, lab: Image.Image):
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(1, int(img.width * s))
        new_h = max(1, int(img.height * s))
        img = img.resize((new_w, new_h), RESIZE_BILINEAR)
        lab = lab.resize((new_w, new_h), RESIZE_NEAREST)
        return img, lab

    def _random_crop_pad_class_aware(self, img: Image.Image, lab: Image.Image):
        import torchvision.transforms as T
        H, W = self.crop_hw

        # pad
        pad_h = max(0, H - img.height)
        pad_w = max(0, W - img.width)
        if pad_h or pad_w:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            lab = TF.pad(lab, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

        best = None
        best_cnt = -1
        for _ in range(max(1, self.crop_retry)):
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(H, W))
            img_c = TF.crop(img, i, j, h, w)
            lab_c = TF.crop(lab, i, j, h, w)
            m = np.array(lab_c, dtype=np.uint8)

            # 희귀 클래스 픽셀 합
            cnt = 0
            for cid in self.rare_ids:
                cnt += int((m == cid).sum())

            if cnt > best_cnt:
                best = (img_c, lab_c)
                best_cnt = cnt
            if cnt >= self.min_rare_px:
                return img_c, lab_c

        return best  # 재시도 내 만족 실패 -> 최선 반환

    def _hflip(self, img: Image.Image, lab: Image.Image):
        if random.random() < 0.5:
            img = TF.hflip(img)
            lab = TF.hflip(lab)
        return img, lab

    # ---------- photometric aug (이미지에만) ----------
    def _apply_photometric(self, img: Image.Image, p: float):
        if not self.args.photo_aug:
            return img
        if random.random() >= p:
            return img

        # gamma (밝기 근사)
        g0, g1 = self.args.gamma_range
        gamma = random.uniform(g0, g1)
        img = ImageEnhance.Brightness(img).enhance(gamma)

        # jitter: 밝기/대비/채도
        jb, jc, js = self.args.jitter
        if jb > 0: img = ImageEnhance.Brightness(img).enhance(random.uniform(1 - jb, 1 + jb))
        if jc > 0: img = ImageEnhance.Contrast(img).enhance(random.uniform(1 - jc, 1 + jc))
        if js > 0: img = ImageEnhance.Color(img).enhance(random.uniform(1 - js, 1 + js))

        # blur
        b0, b1 = self.args.blur_sigma
        sigma = random.uniform(b0, b1)
        if sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        # haze (간단 상수 근사)
        hb0, hb1 = self.args.haze_beta
        if random.random() < 0.3:
            beta = random.uniform(hb0, hb1)
            A = 220
            np_img = np.array(img).astype(np.float32)
            t = np.exp(-beta)
            np_img = np_img * t + A * (1 - t)
            img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))

        # rain (간단 점 오버레이)
        if random.random() < self.args.rain_p:
            np_img = np.array(img).astype(np.float32)
            H, W, _ = np_img.shape
            n = (H * W) // 400
            ys = np.random.randint(0, H, size=n)
            xs = np.random.randint(0, W, size=n)
            np_img[ys, xs, :] = np.clip(np_img[ys, xs, :] + 70, 0, 255)
            img = Image.fromarray(np_img.astype(np.uint8))

        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, lp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        lab = Image.open(lp).convert("L")

        # geometric
        img, lab = self._random_scale(img, lab)
        img, lab = self._random_crop_pad_class_aware(img, lab)
        img, lab = self._hflip(img, lab)

        # photometric (강증강 매칭 시 확률 상향)
        is_strong = False
        if self.strong_keys:
            sp = str(ip)
            is_strong = any(k in sp for k in self.strong_keys)
        p = (self.args.photo_strong_p if is_strong else self.args.photo_p)
        img = self._apply_photometric(img, p=p)

        # to tensor
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
        lab_np = np.array(lab, dtype=np.uint8)
        lab_t = torch.from_numpy(lab_np).long()

        return img_t, lab_t, str(ip)


# ===================== Sampler =====================
def build_sampler_fast(dataset: SegDataset, strong_txt: str, strong_weight: float):
    if not os.path.isfile(strong_txt):
        print(f"[Sampler] strong list not found: {strong_txt}")
        dataset.strong_keys = []
        return None

    with open(strong_txt, "r", encoding="utf-8") as f:
        strong_keys = [ln.strip() for ln in f if ln.strip()]

    # photometric에서도 쓸 수 있게 dataset에 보관
    dataset.strong_keys = strong_keys
    dataset.args._strong_keys = strong_keys  # 다른 곳에서도 접근 가능

    # 문자열 경로 매칭
    img_paths = [str(ip) for (ip, _) in dataset.pairs]
    weights, hit = [], 0
    for p in img_paths:
        w = 1.0
        for k in strong_keys:
            if k in p:
                w = strong_weight
                hit += 1
                break
        weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    print(f"[Sampler] strong hits: {hit}/{len(weights)} ({hit/len(weights)*100:.1f}%)")
    return sampler


# ===================== Eval (val loss) =====================
def evaluate(model, dl, device, amp=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    tot_loss, tot_b = 0.0, 0
    with torch.no_grad():
        it = tqdm(dl, desc="[Eval]", leave=False)
        for imgs, labels, _ in it:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(imgs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = loss_fn(logits, labels)
            b = imgs.size(0)
            tot_loss += loss.item() * b
            tot_b += b
    return tot_loss / max(1, tot_b)


# ===================== Train =====================
def train(args):
    set_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)

    # Dataset/Loader
    train_ds = SegDataset(args.dataset_dir, args.train_split_name, tuple(args.crop_size), tuple(args.scale_range), args, verbose=True)
    val_ds   = SegDataset(args.dataset_dir, args.val_split_name,   tuple(args.crop_size), tuple(args.scale_range), args, verbose=True)

    if args.weighted_sampling and args.strong_list_txt:
        print(f"[Sampler] weighted ON | strong_weight={args.strong_weight} | file={args.strong_list_txt}")
        sampler = build_sampler_fast(train_ds, args.strong_list_txt, args.strong_weight)
        train_dl = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=args.prefetch_factor, drop_last=True
        )
    else:
        print("[Sampler] weighted OFF")
        train_dl = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=args.prefetch_factor, drop_last=True
        )

    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDRNet(num_classes=args.num_classes)
    model = nn.DataParallel(model).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("[Info] channels_last enabled")

    # Load
    if args.loadpath and os.path.isfile(args.loadpath):
        state = torch.load(args.loadpath, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[Load] loaded weights from {args.loadpath}")

    # Optim & Sched
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    steps_per_epoch = max(1, len(train_dl))
    total_steps = args.epochs * steps_per_epoch
    warmup = max(5, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Aug dump
    dump_dir = args.dump_aug_dir
    dump_max = max(0, int(args.dump_aug_n)) if args.dump_aug_n else 0
    dump_count = 0
    if dump_dir:
        os.makedirs(os.path.join(dump_dir, "img"), exist_ok=True)
        os.makedirs(os.path.join(dump_dir, "mask"), exist_ok=True)
        print(f"[AugDump] enabled -> dir={dump_dir}, max={dump_max}")

    print(f"[Run] lr={args.lr} epochs={args.epochs} dataset={args.dataset_dir}")
    best_val = 1e9
    global_seen = 0

    # log header
    with open(os.path.join(args.result_dir, "log.txt"), "w") as f:
        f.write("Epoch\tTrainLoss\tValLoss\tLR\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        it = tqdm(train_dl, desc=f"[GPU 0] Epoch [{epoch}/{args.epochs}]", ncols=100)
        run_loss, seen = 0.0, 0

        for imgs, labels, _ in it:
            # Aug dump (ignore=255 안전)
            if dump_dir and dump_count < dump_max:
                b = imgs.size(0)
                take = min(b, dump_max - dump_count)
                for i in range(take):
                    img_u8 = inv_normalize(imgs[i]).permute(1, 2, 0).numpy()
                    Image.fromarray(img_u8).save(os.path.join(dump_dir, "img", f"{dump_count:05d}.jpg"), quality=95)
                    m = labels[i].cpu().numpy().astype(np.uint8)
                    Image.fromarray(colorize_mask(m)).save(os.path.join(dump_dir, "mask", f"{dump_count:05d}.jpg"), quality=95)
                    dump_count += 1

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = imgs.size(0)
            run_loss += loss.item() * bs
            seen += bs
            global_seen += bs
            it.set_postfix(avg=f"{run_loss / max(1, seen):.4f}",
                           loss=f"{loss.item():.2f}",
                           lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # eval
        val_loss = evaluate(model, val_dl, device, amp=args.amp)

        # log + ckpt
        with open(os.path.join(args.result_dir, "log.txt"), "a") as f:
            f.write(f"{epoch}\t{run_loss/max(1,seen):.4f}\t{val_loss:.4f}\t{optimizer.param_groups[0]['lr']:.6f}\n")

        torch.save(model.module.state_dict(), os.path.join(args.result_dir, f"epoch_{epoch:03d}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.module.state_dict(), os.path.join(args.result_dir, "model_best.pth"))

    print("[Done] best val loss =", best_val)


# ===================== CLI =====================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--loadpath", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--crop_size", type=int, nargs=2, default=[1024, 1024])
    ap.add_argument("--scale_range", type=float, nargs=2, default=[0.5, 2.0])
    ap.add_argument("--train_split_name", type=str, default="train")
    ap.add_argument("--val_split_name", type=str, default="val")

    # weighted sampler
    ap.add_argument("--weighted_sampling", action="store_true")
    ap.add_argument("--strong_list_txt", type=str, default=None)
    ap.add_argument("--strong_weight", type=float, default=3.0)

    # photometric aug
    ap.add_argument("--photo_aug", action="store_true")
    ap.add_argument("--photo_p", type=float, default=0.3)
    ap.add_argument("--photo_strong_p", type=float, default=0.6)
    ap.add_argument("--gamma_range", type=float, nargs=2, default=[0.7, 1.4])
    ap.add_argument("--blur_sigma", type=float, nargs=2, default=[0.0, 1.2])
    ap.add_argument("--jitter", type=float, nargs=3, default=[0.2, 0.2, 0.2])  # brightness/contrast/saturation
    ap.add_argument("--haze_beta", type=float, nargs=2, default=[0.02, 0.06])
    ap.add_argument("--rain_p", type=float, default=0.15)

    # class-aware crop
    ap.add_argument("--rare_ids", type=int, nargs="+", default=[7, 11, 12, 13, 9, 10, 16])
    ap.add_argument("--crop_retry", type=int, default=10)
    ap.add_argument("--min_rare_px", type=int, default=800)

    # runtime
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # aug dump
    ap.add_argument("--dump_aug_dir", type=str, default=None)
    ap.add_argument("--dump_aug_n", type=int, default=60)
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

