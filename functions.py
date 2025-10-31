# functions.py
import os
import ast
import csv
import math
import random
from glob import glob
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
import torchvision.transforms.functional as TF

# -----------------------------
# 기본 유틸
# -----------------------------
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f'Argument "{s}" is not a list')
    return v

def display_dataset_info(datadir, dataset=None):
    print(f"Dataset path: {datadir}")
    if dataset is not None:
        print(f"Found {len(dataset)} images.")

def load_state_dict(model, state_dict, strict=False):
    """
    model.module / model 키 프리픽스 차이를 자동 보정
    """
    new_state = OrderedDict()
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    for k, v in state_dict.items():
        if is_ddp and not k.startswith("module."):
            k = "module." + k
        if (not is_ddp) and k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    mkeys = set(model.state_dict().keys())
    lkeys = set(new_state.keys()) & mkeys
    total = len(mkeys); loaded = len(lkeys)
    print(f"[Info] Loaded {loaded}/{total} state_dict entries ({100.0*loaded/max(1,total):.2f}%) from checkpoint.")
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)}")

# -----------------------------
# 클래스 가중치 계산
# -----------------------------
def compute_class_weights(
    label_dir,
    num_classes=19,
    ignore_index=255,
    mode="median",         # "median" | "effective" | "inverse"
    normalize=True,
    clip=None,
    beta=0.9999,           # for effective number
):
    counts = np.zeros(num_classes, dtype=np.int64)
    paths = glob(os.path.join(label_dir, "**", "*.png"), recursive=True)
    for p in paths:
        y = np.array(Image.open(p))
        m = (y != ignore_index)
        if m.any():
            binc = np.bincount(y[m].ravel(), minlength=num_classes)
            counts += binc[:num_classes]

    counts = np.maximum(counts, 1)

    if mode == "median":
        freq = counts / counts.sum()
        med = np.median(freq[freq > 0])
        w = med / np.clip(freq, 1e-12, None)
    elif mode == "effective":
        eff = (1 - np.power(beta, counts)) / (1 - beta)
        w = 1.0 / eff
    elif mode == "inverse":
        w = 1.0 / counts
    else:
        raise ValueError("mode must be one of ['median','effective','inverse']")

    if normalize:
        w = w * (len(w) / w.sum())

    if clip is not None:
        w = np.clip(w, clip[0], clip[1])

    return torch.tensor(w, dtype=torch.float32), counts

# -----------------------------
# 손실 함수
# -----------------------------
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None, aux_weights=(1.0, 0.4)):
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds, labels):
        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, (list, tuple)):
            total = 0.0
            for w, p in zip(self.aux_weights, preds):
                if w != 0.0:
                    total += w * self._forward(p, labels)
            return total
        return self._forward(preds, labels)

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None, thresh=0.6, aux_weights=(1.0, 0.4)):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds, labels):
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

    def forward(self, preds, labels):
        if isinstance(preds, (list, tuple)):
            total = 0.0
            for w, p in zip(self.aux_weights, preds):
                if w != 0.0:
                    total += w * self._forward(p, labels)
            return total
        return self._forward(preds, labels)

class FocalCELoss(nn.Module):
    """
    train.py에서 FocalCELoss(gamma=...) 로 호출해도 되도록 시그니처 포함
    """
    def __init__(self, gamma=2.0, weight=None, ignore_index=255, aux_weights=(1.0, 0.3)):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux_weights = aux_weights

    def _prep_weight(self, logits):
        if self.weight is None:
            return None
        if isinstance(self.weight, torch.Tensor):
            w = self.weight.to(device=logits.device, dtype=logits.dtype)
        elif isinstance(self.weight, (list, tuple, np.ndarray)):
            w = torch.tensor(self.weight, device=logits.device, dtype=logits.dtype)
        else:
            raise ValueError("weight must be None, list/tuple/ndarray, or torch.Tensor")
        if w.numel() != logits.size(1):
            raise ValueError(f"weight length ({w.numel()}) must equal num classes ({logits.size(1)})")
        return w

    def _focal_ce(self, logits, target):
        w = self._prep_weight(logits)
        ce = F.cross_entropy(logits, target, weight=w, ignore_index=self.ignore_index, reduction='none')
        mask = (target != self.ignore_index).float()
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def forward(self, preds, target):
        if isinstance(preds, (list, tuple)):
            total = 0.0
            for i, p in enumerate(preds):
                wi = self.aux_weights[i] if i < len(self.aux_weights) else 0.0
                if wi != 0.0:
                    total = total + wi * self._focal_ce(p, target)
            return total
        else:
            return self._focal_ce(preds, target)

# -----------------------------
# 스케줄러
# -----------------------------
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * float(self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

class EpochWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', total_epochs=500, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self._get_lr_ratio()
        return [max(ratio * lr, 1e-7) for lr in self.base_lrs]

    def _get_lr_ratio(self):
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_ratio()
        return self._get_main_ratio()

    def _get_warmup_ratio(self):
        alpha = self.last_epoch / max(1, self.warmup_epochs)
        if self.warmup == 'linear':
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        else:
            return self.warmup_ratio ** (1. - alpha)

    def _get_main_ratio(self):
        raise NotImplementedError

class WarmupPolyEpochLR(EpochWarmupLR):
    def __init__(self, optimizer, power=0.9, total_epochs=500, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_epochs, warmup_ratio, warmup, total_epochs, last_epoch)

    def _get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epochs
        real_total = max(1, self.total_epochs - self.warmup_epochs)
        alpha = min(max(real_epoch / real_total, 0.0), 1.0)
        return (1 - alpha) ** self.power

# -----------------------------
# 증강(훈련/검증)
# -----------------------------
class SegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.75, 1.25]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest  = transforms.InterpolationMode.NEAREST

    def __call__(self, image, label):
        # Random scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = image.size
        nh, nw = int(h * scale), int(w * scale)
        image = TF.resize(image, (nh, nw), interpolation=self.bilinear)
        label = TF.resize(label, (nh, nw), interpolation=self.nearest)

        # Pad to crop
        ph = max(self.crop_size[0] - nh, 0)
        pw = max(self.crop_size[1] - nw, 0)
        if ph > 0 or pw > 0:
            image = TF.pad(image, (0,0,pw,ph), fill=0)
            label = TF.pad(label, (0,0,pw,ph), fill=255)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # H-flip
        if random.random() > 0.5:
            image = TF.hflip(image); label = TF.hflip(label)

        # To tensor & normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        return image, label

class ValTransform:
    def __init__(self, crop_size=[1024,1024]):
        self.crop_size = crop_size
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest  = transforms.InterpolationMode.NEAREST

    def __call__(self, image, label):
        # center crop or resize to crop_size
        image = TF.resize(image, self.crop_size, interpolation=self.bilinear)
        label = TF.resize(label, self.crop_size, interpolation=self.nearest)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        return image, label

# -----------------------------
# 견고한 라벨 경로 추정 + 페어 수집
# -----------------------------
def _guess_label_path_from_image(image_path: str, root_dir: str) -> str:
    """
    image_path: .../image/<split>/<subdir>/FILE.(jpg|png|jpeg|...)
    return:     .../labelmap/<split>/<subdir>/FILE_...CategoryId.png (실존 파일 우선)
    규칙:
      - 파일명에 "_leftImg8bit"가 있으면 "_gtFine_CategoryId.png"로.
      - 아니면 "_CategoryId.png".
      - 여러 후보를 순차적으로 검사하며, 마지막엔 폴더 스캔 폴백.
    """
    image_dir = os.path.join(root_dir, "image")
    label_dir = os.path.join(root_dir, "labelmap")

    rel = os.path.relpath(image_path, image_dir)          # <split>/.../FILE.ext
    parts = rel.split(os.sep)
    split = parts[0]
    subparts = parts[1:]
    fname = subparts[-1]
    name, _ext = os.path.splitext(fname)

    candidates = []
    if "_leftImg8bit" in name:
        base = name.replace("_leftImg8bit", "_gtFine_CategoryId")
        candidates += [base + ".png", base + ".PNG"]
    else:
        candidates += [name + "_CategoryId.png", name + "_CategoryId.PNG"]

    # 보조 후보: _leftImg8bit 제거 후 gtFine 접미사
    if "_gtFine_CategoryId" not in candidates[0]:
        candidates.append(name.replace("_leftImg8bit", "") + "_gtFine_CategoryId.png")

    # 경로 조립 & 존재 확인
    subdirs = subparts[:-1]
    for cf in candidates:
        lp = os.path.join(label_dir, split, *subdirs, cf)
        if os.path.isfile(lp):
            return lp

    # 폴백: 폴더 내 stem이 같은 *_CategoryId.png 검색
    folder = os.path.join(label_dir, split, *subdirs)
    if os.path.isdir(folder):
        stem = name.replace("_leftImg8bit", "")
        for f in os.listdir(folder):
            if f.endswith("CategoryId.png") and f.startswith(stem):
                return os.path.join(folder, f)

    # 실패 시 첫 후보 반환(호출부에서 존재 여부 다시 판단)
    return os.path.join(label_dir, split, *subdirs, candidates[0])

class SegmentationDataset(torch.utils.data.Dataset):
    """
    이미지-라벨 페어를 '실존하는 것만' 수집.
    """
    def __init__(self, root_dir, crop_size, subset, scale_range, transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform if transform is not None else SegmentationTransform(crop_size, scale_range)

        img_glob = os.path.join(root_dir, "image", subset, "*", "*.*")
        all_imgs = sorted(glob(img_glob, recursive=True))

        self.pairs = []
        missing = 0
        for ip in all_imgs:
            lp = _guess_label_path_from_image(ip, root_dir)
            if os.path.isfile(lp):
                self.pairs.append((ip, lp))
            else:
                missing += 1
        if missing > 0:
            print(f"[SegDataset] subset={subset} 페어 누락 {missing}개 제외 (남은 {len(self.pairs)})")
        if len(self.pairs) == 0:
            raise RuntimeError(f"[SegDataset] subset={subset} 유효한 이미지-라벨 페어가 없습니다.")

        # 필요시 라벨 매핑 테이블 (그대로 유지)
        self.label_map = np.arange(256, dtype=np.uint8)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, lp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        lab = Image.open(lp).convert("L")
        img, lab = self.transform(img, lab)
        if not isinstance(lab, torch.Tensor):
            lab = torch.from_numpy(np.array(lab, dtype=np.uint8))
        return img, lab.long()

