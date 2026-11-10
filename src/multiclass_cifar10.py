"""
Long-Tailed CIFAR-10 (CIFAR-10-LT) Imbalanced Learning Benchmark (MiSLAS Stage-2 Family)
based on Hu et al. (ICML 2025), "Learning Imbalanced Data with Beneficial Label Noise".

Evaluates a ResNet-32 backbone loaded from the official pre-trained MiSLAS Stage-1
checkpoint on CIFAR-10-LT (IR = 100, N_max/N_min = 5000 / 50).

Protocol: by default every method is evaluated once, after the last Stage-2 epoch (no epoch
selection on test data); `--epoch_selection best_test` instead keeps the epoch with the highest
test accuracy, as in the LNR authors' code. `--seeds` repeats the runs; results per seed and
mean ± std over the seeds are written to `--out_dir` (default: a new, automatically named folder
per run, results/cifar10_<protocol>_seeds<seeds>[_<changed options>]_<date>_<time>).

SelMix's 10k validation set uses 500 unused CIFAR-10 training images per class for classes 1-9
and 500 class-0 test images (plus flipped copies); those class-0 test images are removed, and
all methods are evaluated on the remaining 9,500 test images (`build_selmix_val_set`).

Codes that are implemented from the following papers or adapted from their existing implementations:
    1. Stage-2 Decoupled Fine-Tuning Protocol & Checkpoint Loading:
       Adapted from Zhong et al. (CVPR 2021), "Improving Calibration for Long-Tailed
       Recognition", official repository:
       https://github.com/JIA-Lab-research/MiSLAS/blob/main/train_stage2.py
    2. LNR Stage-2 Integration (`MiSLAS+LNR`):
       Implemented from Hu et al. (ICML 2025), "Learning Imbalanced Data with
       Beneficial Label Noise", Section 4.3, Section 5, and Appendix C.2:
       https://github.com/guangzhengh/LNR
    3. Selective Mixup fine-tuning (`SelMix`, `MiSLAS+SelMix(10k)`):
       Ramasubramanian et al. (ICLR 2024), "Selective Mixup Fine-Tuning for Optimizing
       Non-Decomposable Metrics", with SelMix's own supervised defaults; ported from the
       official repository https://github.com/val-iisc/SelMix (see the docstrings of
       `build_selmix_val_set`, `SelMix`, `freeze_bn_selmix`, `selmix_optimizer`, `selmix_finetune`).
"""

import argparse
import math
import random
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Resolves project root: src/.. -> DD2610_final_project_Sep2026/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
data_root = PROJECT_ROOT / "data"

from utils.cifar_utils import (
    CIFAR10_LT,
    ClassBalancedSampler,
    LabelAwareSmoothing,
    ResNet32,
    compute_ece,
    compute_metrics,
    compute_train_posteriors,
    get_device,
    lnr_multiclass_flip,
)


# ---------------------------------------------------------------------------
# Checkpoint Loader for Official MiSLAS Stage-1 Weights
# ---------------------------------------------------------------------------

DEFAULT_STAGE1_CHECKPOINT = data_root / "cifar10" / "cifar10_imb001_stage1_mixup10.pth.tar"

def load_mislas_stage1_checkpoint(
    model: ResNet32,
    checkpoint_path: str = str(DEFAULT_STAGE1_CHECKPOINT),
    device: torch.device = None,
):
    """
    Loads an official MiSLAS Stage-1 checkpoint (.pth.tar) into ResNet32. Default: the
    CIFAR-10-LT (IF=100) Stage-1 (mixup) model released in the MiSLAS README, saved under
    data/cifar10 (trained with CIFAR-10-LT subsampling seed 0, i.e. `--lt_seed 0`).

    Source:
        Adapted from `train_stage2.py` in Zhong et al. (CVPR 2021):
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/train_stage2.py
    """
    device = device if device is not None else get_device()
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found at '{ckpt_file}'.\n"
            f"Expected cifar10_imb001_stage1_mixup10.pth.tar under {DEFAULT_STAGE1_CHECKPOINT.parent}, "
            "or pass another path via --stage1_checkpoint <path>."
        )

    print(f"Loading pre-trained Stage-1 checkpoint from: {ckpt_file}")
    # weights_only=False: MiSLAS checkpoints also pickle a NumPy scalar (`its_ece`), which the
    # weights-only loader (the default since PyTorch 2.6) refuses; only load trusted files
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    # Official MiSLAS checkpoints store backbone weights under 'state_dict_model' (or 'state_dict')
    # and the stage-1 classifier under 'state_dict_classifier'
    if isinstance(checkpoint, dict):
        state_dict = dict(checkpoint.get("state_dict_model", checkpoint.get("state_dict", checkpoint)))
        state_dict.update(checkpoint.get("state_dict_classifier", {}))
        print(f"Checkpoint epoch: {checkpoint.get('epoch')}, best_acc1: {checkpoint.get('best_acc1')}")
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        # Map official MiSLAS `linear.weight` / `linear.bias` to `fc.weight` / `fc.bias`
        if new_key.startswith("linear."):
            new_key = new_key.replace("linear.", "fc.")
        cleaned_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    # `lws.learned_norm` is newly initialized in Stage 2 and expected to be in `missing`
    non_lws_missing = [k for k in missing if not k.startswith("lws.")]
    if non_lws_missing:
        print(f"Warning: Missing keys when loading Stage-1 checkpoint: {non_lws_missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in Stage-1 checkpoint: {unexpected}")


def configure_mislas_stage2(model: ResNet32):
    """
    MiSLAS Stage-2 decoupling (classifier head tuning).

    Source:
        Adapted from `train_stage2.py` in Zhong et al. (CVPR 2021):
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/train_stage2.py
        As in MiSLAS, the whole backbone is frozen, including the Batch Normalization
        scale and shift; only the classifier (`fc`) and Learnable Weight Scaling (`lws`)
        are trained. Whether the BN running statistics are updated (shifted BN) is set by
        `--shift_bn` through the train/eval mode in `train_epoch`.
    """
    for name, param in model.named_parameters():
        if "fc" in name or "lws" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Auxiliary Method Modules
# ---------------------------------------------------------------------------
class ArrayDataset(Dataset):
    """Images (N, H, W, C, uint8) with labels; returns (image, label) like torchvision's CIFAR10."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        self.data = data
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


def build_selmix_val_set(
    data_dir: str,
    train_set: CIFAR10_LT,
    test_set: datasets.CIFAR10,
    transform,
    per_class: int = 500,
    seed: int = 0,
):
    """
    SelMix's 10k validation set, built so that it contains no image the model is trained on
    and no image it is evaluated on.

    SelMix's own construction holds out 500 images per class from the CIFAR-10 training split
    and adds a horizontally flipped copy of each image, giving 10,000 validation images.
    Here, the 500 images per class must also be unused by CIFAR-10-LT, on which the MiSLAS
    stage-1 checkpoint was trained:
      - classes 1-9: 500 CIFAR-10 training images that are not in CIFAR-10-LT (drawn with `seed`);
      - class 0: CIFAR-10-LT uses all 5000 class-0 training images, so 500 class-0 test images
        are used instead; they are removed from the evaluation set of ALL methods.

    Returns (val_set, eval_indices): the validation dataset (with `transform`) and the indices of
    the test images left for evaluation (9,500).

    Reference:
        The held-out 500 images per class plus horizontally flipped copies follow
        `LT_Dataset.resize` in Ramasubramanian et al. (ICLR 2024), official repository:
        https://github.com/val-iisc/SelMix/blob/main/datasets/cifar.py
        The choice of unused training images and of class-0 test images is ours.
    """
    rng = np.random.RandomState(seed)
    base_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    train_targets = np.array(base_train.targets)
    test_targets = np.array(test_set.targets)
    used = set(np.asarray(train_set.indices).tolist())

    val_data, val_targets, test_val_idx = [], [], []
    for c in range(10):
        unused = np.array([i for i in np.where(train_targets == c)[0] if i not in used], dtype=int)
        if len(unused) >= per_class:
            idx = rng.choice(unused, per_class, replace=False)
            val_data.append(base_train.data[idx])
            source = f"{per_class} of {len(unused)} unused training images"
        else:
            idx = rng.choice(np.where(test_targets == c)[0], per_class, replace=False)
            val_data.append(test_set.data[idx])
            test_val_idx.extend(idx.tolist())
            n_test = int((test_targets == c).sum())
            source = f"{per_class} of {n_test} test images (removed from evaluation) (no unused training images)"
        val_targets += [c] * per_class
        print(f"SelMix validation, class {c}: {source}")

    val_data = np.concatenate(val_data)
    val_targets = np.array(val_targets)
    # Horizontally flipped copies (flip along the width axis of N x H x W x C)
    val_data = np.concatenate([val_data, val_data[:, :, ::-1, :]])
    val_targets = np.concatenate([val_targets, val_targets])

    eval_idx = np.setdiff1d(np.arange(len(test_targets)), np.array(test_val_idx, dtype=int))
    return ArrayDataset(val_data, val_targets, transform), eval_idx


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def gpu_augment(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, augment: bool = True) -> torch.Tensor:
    """
    The training transform of this script on a batch of uint8 images (B, 32, 32, 3) that are
    already on the device: RandomCrop(32, padding=4) (zero padding) + RandomHorizontalFlip +
    ToTensor + Normalize, as tensor operations (one random crop offset and flip per image),
    instead of PIL image by image on the CPU. With augment=False only ToTensor + Normalize.
    Returns a float tensor (B, 3, 32, 32).
    """
    x = images.permute(0, 3, 1, 2).float() / 255.0  # ToTensor: (B, 3, 32, 32) in [0, 1]
    if augment:
        b, device = x.size(0), x.device
        x = F.pad(x, (4, 4, 4, 4))  # zero padding, as RandomCrop(padding=4, fill=0)
        top = torch.randint(0, 9, (b,), device=device)
        left = torch.randint(0, 9, (b,), device=device)
        ar = torch.arange(32, device=device)
        rows = (top[:, None] + ar)[:, :, None]  # (B, 32, 1)
        cols = (left[:, None] + ar)[:, None, :]  # (B, 1, 32)
        x = x[torch.arange(b, device=device)[:, None, None], :, rows, cols]  # (B, 32, 32, 3)
        x = x.permute(0, 3, 1, 2)
        flip = torch.rand(b, device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return ((x - mean[:, None, None]) / std[:, None, None]).contiguous()


class SelMix:
    """
    SelMix fine-tuning (selective mixup) with the min-recall objective.

    Every `refresh_every` SGD steps, the sampling distribution P(y1, y2) over class pairs is
    re-estimated on the validation set: from the validation confusion matrix and the class
    prototypes (mean validation features), SelMix estimates how much a mixup of a class-y1
    feature with a class-y2 feature (label y1) would raise the objective (gain matrix G), and
    samples class pairs with P = softmax(G / temperature), restricted to pairs with a positive
    gain. The mixed pairs are drawn from the TRAINING set; the validation set is only used to
    estimate P.

    Hyper-parameters (SelMix defaults for supervised CIFAR-10-LT, rho = 100):
        temperature 0.1 (gain scaling s = 10, Table G.2), min-recall multiplier step
        val_lr = 50 (omega_MinRec, Table G.2), beta = 0 and lambda_min = 0.6 (official config),
        refresh every 50 steps ("a validation step every 50 steps", App. L).

    Source / Attribution:
        Ramasubramanian et al. (ICLR 2024), "Selective Mixup Fine-Tuning for Optimizing
        Non-Decomposable Metrics", official repository https://github.com/val-iisc/SelMix:
        - `update` ports `SelMix.val` and `SelMix.feedforward` (finetuning/selmix_ssl.py),
          `MinRecall` (MetricOptimisation/metrics.py: `lambda_update`, `metric_gain_rate`,
          `SamplingDistribution`) and `MetricOptimizer` (MetricOptimisation/base_optimizer.py:
          `mixups`, `inference`, `logit_change_rate`), vectorised with torch.einsum.
        - `sample_batch` ports `FastJointSampler` (dataloaders/sampler.py).
        - Hyper-parameter values: Table G.2 of the paper and
          configs/cifar10/N1-1500_M1-3000_IBRL-100_IBRU-100/MinRecall.yaml.
        The same code is used for MiSLAS+SelMix in https://github.com/guangzhengh/LNR.
    """

    def __init__(
        self,
        train_set: CIFAR10_LT,
        val_set,
        num_classes: int = 10,
        lambda_min: float = 0.6,
        temperature: float = 0.1,
        beta: float = 0.0,
        val_lr: float = 50.0,
        refresh_every: int = 50,
        device: torch.device = None,
    ):
        self.device = device if device is not None else get_device()
        self.train_set = train_set
        self.val_set = val_set
        self.num_classes = num_classes
        self.lambda_min = lambda_min
        self.temperature = temperature
        self.beta = beta
        self.val_lr = val_lr
        self.refresh_every = refresh_every
        self.lambdas = np.full(num_classes, 1.0 / num_classes)  # Lagrange multipliers of min recall
        targets = np.asarray(train_set.targets)
        self.class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        self.P = np.full((num_classes, num_classes), 1.0 / num_classes ** 2)

        # Training and validation images are kept on the device (uint8, ~38 MB and ~31 MB), so
        # batches are built and augmented there (`gpu_augment`) instead of image by image on the CPU
        self.train_images = torch.from_numpy(np.asarray(train_set.data)).to(self.device)
        self.val_images = torch.from_numpy(np.asarray(val_set.data)).to(self.device)
        self.val_targets = torch.as_tensor(np.asarray(val_set.targets), dtype=torch.long, device=self.device)
        self.mean = torch.tensor(CIFAR_MEAN, device=self.device)
        self.std = torch.tensor(CIFAR_STD, device=self.device)
        # Training indices grouped by class, for drawing one image of a given class per sample
        self.sorted_idx = np.concatenate(self.class_indices)
        self.class_sizes = np.array([len(idx) for idx in self.class_indices])
        self.class_offsets = np.concatenate([[0], np.cumsum(self.class_sizes)[:-1]])

    @torch.no_grad()
    def update(self, model: ResNet32, device: torch.device = None):
        """Re-estimates the sampling distribution self.P on the validation set."""
        device = device if device is not None else get_device()
        K = self.num_classes
        was_training = model.training
        model.eval()
        feats, preds, labels = [], [], []
        for start in range(0, len(self.val_targets), 1000):  # validation images already on the device
            inputs = gpu_augment(self.val_images[start:start + 1000], self.mean, self.std, augment=False)
            _, feat = model(inputs)
            feats.append(feat)
            preds.append(model.fc(feat).argmax(dim=1))  # classifier logits (no weight scaling)
            labels.append(self.val_targets[start:start + 1000])
        model.train(was_training)
        feats, preds, labels = torch.cat(feats), torch.cat(preds), torch.cat(labels)

        # Joint confusion matrix CM[i, j] = P(y = i, y_hat = j) and class prototypes
        CM = torch.zeros(K, K, device=device)
        CM.index_put_((labels, preds), torch.ones_like(labels, dtype=torch.float), accumulate=True)
        CM = (CM / len(labels)).clamp(1e-7, 1e4)
        prototypes = torch.stack([feats[labels == c].mean(dim=0) for c in range(K)])  # (K, D)

        # Min recall: multiplier update, then dM/dA[i, j], the gain rate of the objective
        # when logit j of samples of class i increases
        recall = (torch.diag(CM) / CM.sum(dim=1)).cpu().numpy()
        lambdas = self.lambdas ** self.beta * np.exp(-self.val_lr * recall)
        self.lambdas = lambdas / lambdas.sum()
        diag = torch.diag(CM)
        dM_dA = -K * CM * diag.unsqueeze(1)  # i != j: -K * CM[i, j] * CM[i, i]
        ar = torch.arange(K, device=device)
        dM_dA[ar, ar] = diag - K * diag ** 2  # i == j: CM[i, i] - K * CM[i, i]^2
        dM_dA = torch.tensor(self.lambdas, dtype=torch.float32, device=device).unsqueeze(1) * K * dM_dA

        # Mixed prototypes MP[i, j], their predictions, and the logit change rate
        # LCR[i, j, q, l]: change of logit l on prototype q after an i-j mixup
        m = (self.lambda_min + 1.0) / 2.0
        MP = m * prototypes[:, None, :] + (1.0 - m) * prototypes[None, :, :]  # (K, K, D)
        mixup_pred = F.softmax(model.fc(MP), dim=-1)  # (K, K, K)
        dot = torch.einsum("ijd,qd->ijq", MP, prototypes)  # MP[i, j] . prototype_q
        LCR = -dot[:, :, :, None] * mixup_pred[:, :, None, :]  # (K, K, K, K)
        for i in range(K):
            LCR[i, :, :, i] += dot[i, i, :]  # as written in the official `logit_change_rate`

        # Gain of each i-j mixup and the sampling distribution over pairs with a positive gain
        G = torch.einsum("ql,ijql->ij", dM_dA, LCR).cpu().numpy()
        mask = (G > 0).astype(float)
        P = np.exp((G - G.max()) / self.temperature)
        P = P / P.sum() * mask
        self.P = (P + 1e-8) / np.sum(P + 1e-8)

    def sample_batch(self, batch_size: int, device: torch.device = None):
        """Draws (x1, y1, x2) from the training set: y1 ~ P(y1), y2 ~ P(y2 | y1), images uniform per class."""
        device = device if device is not None else get_device()
        K = self.num_classes
        eps = 1.0 / K ** 3
        p_y1 = np.sum(self.P + eps, axis=1)
        p_y2_given_y1 = (self.P + eps) / (p_y1[:, None] + eps)
        p_y2_given_y1 = p_y2_given_y1 / p_y2_given_y1.sum(axis=1, keepdims=True)
        y1 = np.random.choice(K, size=batch_size, p=p_y1 / p_y1.sum())
        # y2 | y1 for the whole batch at once (inverse CDF of each sample's row of P(y2 | y1))
        cdf = np.cumsum(p_y2_given_y1[y1], axis=1)
        y2 = np.minimum((np.random.rand(batch_size, 1) > cdf).sum(axis=1), K - 1)
        x1 = gpu_augment(self.train_images[self._draw(y1)], self.mean, self.std)
        x2 = gpu_augment(self.train_images[self._draw(y2)], self.mean, self.std)
        return x1.to(device), torch.as_tensor(y1, dtype=torch.long, device=device), x2.to(device)

    def _draw(self, classes: np.ndarray) -> torch.Tensor:
        """One training image drawn uniformly from each given class (indices into train_images)."""
        r = np.floor(np.random.rand(len(classes)) * self.class_sizes[classes]).astype(int)
        idx = self.sorted_idx[self.class_offsets[classes] + r]
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)


def freeze_bn_selmix(model: nn.Module):
    """
    Batch normalization as in SelMix fine-tuning: the running statistics are no longer updated
    (momentum 0, track_running_stats False). The training forward pass therefore uses batch
    statistics, and evaluation keeps the stage-1 running statistics. The BN scale and shift
    are frozen in `selmix_optimizer`.

    Source:
        `SelMix.train` in https://github.com/val-iisc/SelMix/blob/main/finetuning/selmix_ssl.py
        (bn_momentum 0.0 in the official configs).
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.momentum = 0.0
            module.track_running_stats = False


def selmix_optimizer(model: ResNet32, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Plain SGD with SelMix's parameter groups: backbone at lr / 10 with weight decay, BN scale
    and shift frozen, classifier weight at lr with weight decay, classifier bias at lr without
    weight decay. No momentum is set, as in the official code. The weight scaling `lws` is not
    part of SelMix and stays at 1.

    Source:
        `get_finetune_SGD` (freeze_backbone=False) in
        https://github.com/val-iisc/SelMix/blob/main/train_utils.py, as called by trainMetricOpt.py;
        learning rates from Table G.2 of the paper (classifier 3e-3, backbone 3e-4).
    """
    backbone, fc_weight, fc_bias = [], [], []
    for name, param in model.named_parameters():
        if name.startswith("lws."):
            param.requires_grad = False
        elif name.startswith("fc."):
            param.requires_grad = True
            (fc_bias if name.endswith("bias") else fc_weight).append(param)
        elif "bn" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            backbone.append(param)
    return torch.optim.SGD(
        [
            {"params": backbone, "lr": lr / 10.0},
            {"params": fc_weight, "lr": lr},
            {"params": fc_bias, "lr": lr, "weight_decay": 0.0},
        ],
        lr=lr, weight_decay=weight_decay,
    )


SELMIX_VARIANTS = {
    "frozen-mislasopt": "backbone frozen (BN eval mode); fc trained with MiSLAS's Stage-2 optimizer and epochs",
    "frozen": "backbone frozen (BN eval mode); fc trained with SelMix's optimizer and steps",
    "full": "SelMix's supervised default: backbone fine-tuned (BN batch statistics); SelMix's optimizer and steps",
}


def selmix_finetune(
    model: ResNet32,
    selmix: SelMix,
    args,
    device: torch.device = None,
    eval_fn=None,
    eval_every: int = 50,
    variant: str = "frozen",
    steps_per_epoch: int = None,
):
    """
    Fine-tunes the stage-1 model with SelMix (SelMix's Algorithm 2): every
    `selmix.refresh_every` steps P is re-estimated on the validation set, then each step mixes
    the features of x1 and x2 (lam ~ U(0.6, 1)) and minimizes the cross-entropy on y1.

    The SelMix loss, sampling and validation set are the same in all variants; they differ in
    what is trained and with which optimizer:
      frozen-mislasopt: the backbone is frozen with BN in eval mode, as in MiSLAS Stage 2; only
          fc is trained, with MiSLAS's Stage-2 optimizer (SGD, momentum 0.9, lr_factor x
          stage2_lr, weight_decay, cosine per epoch) for stage2_epochs x steps_per_epoch steps,
          evaluated once per epoch. Closest to the LNR authors' MiSLAS+SelMix code.
      frozen: the backbone is frozen with BN in eval mode; only fc is trained, with SelMix's
          optimizer (plain SGD, selmix_lr, selmix_weight_decay without decay on the bias,
          cosine per step) for selmix_steps steps.
      full: SelMix's own supervised default: the backbone is also fine-tuned (selmix_lr / 10),
          BN scale and shift are frozen but training uses batch statistics (`freeze_bn_selmix`).

    If `eval_fn` is given (best-epoch protocol), it is called every `eval_every` steps
    (frozen-mislasopt: every epoch) and after the last step; returned are the evaluation with
    the highest overall accuracy and the one after the last step, each as
    (metrics, "step k" or "epoch k"). Otherwise None is returned.

    Source / Attribution:
        `SelMix.train` (finetuning/selmix_ssl.py), the feature mixup of `forward` in
        models/wrapper.py, and the CosineAnnealingLR schedule of trainMetricOpt.py in
        https://github.com/val-iisc/SelMix; frozen backbone and MiSLAS optimizer as in
        train_stage2_selmix.py of https://github.com/guangzhengh/LNR.
    """
    device = device if device is not None else get_device()
    if variant == "full":
        freeze_bn_selmix(model)
        optimizer = selmix_optimizer(model, args.selmix_lr, args.selmix_weight_decay)
        total_steps, sched_every, train_backbone, fc_group = args.selmix_steps, 1, True, 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
    else:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
        train_backbone, fc_group = False, 0
        if variant == "frozen":
            optimizer = torch.optim.SGD(
                [{"params": [model.fc.weight]}, {"params": [model.fc.bias], "weight_decay": 0.0}],
                lr=args.selmix_lr, weight_decay=args.selmix_weight_decay,
            )
            total_steps, sched_every = args.selmix_steps, 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
        elif variant == "frozen-mislasopt":
            optimizer = torch.optim.SGD(
                model.fc.parameters(), lr=args.lr_factor * args.stage2_lr, momentum=0.9,
                weight_decay=args.weight_decay,
            )
            total_steps, sched_every = args.stage2_epochs * steps_per_epoch, steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=0.0)
            eval_every = steps_per_epoch
        else:
            raise ValueError(f"Unknown SelMix variant: {variant}")

    def position(step):
        return f"epoch {step // steps_per_epoch}" if variant == "frozen-mislasopt" else f"step {step}"

    running_loss, running_steps = 0.0, 0
    best = None  # (metrics, position) with the highest overall test accuracy, if eval_fn is given
    for step in range(1, total_steps + 1):
        if (step - 1) % selmix.refresh_every == 0:
            selmix.update(model, device)
        x1, y1, x2 = selmix.sample_batch(args.batch_size, device)
        if train_backbone:
            model.train()
            _, feat1 = model(x1)
            _, feat2 = model(x2)
        else:
            model.eval()  # frozen backbone, BN with its stored statistics
            with torch.no_grad():
                _, feat1 = model(x1)
                _, feat2 = model(x2)
        lam = torch.empty(x1.size(0), 1, device=device).uniform_(selmix.lambda_min, 1.0)
        logits = model.fc(lam * feat1 + (1.0 - lam) * feat2)
        loss = F.cross_entropy(logits, y1)  # temperature T = 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % sched_every == 0:
            scheduler.step()

        running_loss += loss.item()
        running_steps += 1
        if step % 250 == 0 or step == total_steps:
            print(
                f"SelMix ({variant}) step [{step:05d}/{total_steps}] | Loss: {running_loss / running_steps:.4f} | "
                f"LR (classifier): {optimizer.param_groups[fc_group]['lr']:.6f}"
            )
            running_loss, running_steps = 0.0, 0

        if eval_fn is not None and (step % eval_every == 0 or step == total_steps):
            metrics = eval_fn()
            if best is None or metrics["Overall"] > best[0]["Overall"]:
                best = (metrics, position(step))
            last = (metrics, position(step))  # after the loop: the evaluation after the last step
    model.eval()
    return (best, last) if eval_fn is not None else None


# ---------------------------------------------------------------------------
# Stage-2 Training Epoch Routine
# ---------------------------------------------------------------------------
def train_epoch(
    model: ResNet32,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    class_counts: list[int],
    method: str = "MiSLAS+LNR",
    epoch: int = 1,
    t_flip: float = 3.5,
    flipped_targets: torch.Tensor = None,
    shift_bn: bool = False,
    device: torch.device = None,
):
    """
    Executes one epoch of MiSLAS Stage-2 classifier fine-tuning.

    Source / Attribution:
        - Base MiSLAS Stage-2 loop adapted from Zhong et al. (CVPR 2021):
          https://github.com/JIA-Lab-research/MiSLAS/blob/main/train_stage2.py
        - LNR label flipping (`MiSLAS+LNR`) from Hu et al. (ICML 2025),
          Algorithm 2 & Appendix C.2.
    """
    device = device if device is not None else get_device()
    # MiSLAS Stage 2: backbone in eval mode unless shift_bn (then BN running statistics are
    # re-estimated on the class-balanced data); fc and lws have no train/eval-specific layers
    if shift_bn:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    num_samples = 0

    for inputs, targets, idx in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if method == "MiSLAS":
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

        elif method == "MiSLAS+LNR":
            logits, _ = model(inputs)
            # Labels flipped once at the start of this epoch on the whole training set (Algorithm 2, round t)
            targets_mod = flipped_targets[idx.to(device)]
            loss = criterion(logits, targets_mod)

        else:
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    return total_loss / num_samples


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_counts: list[int],
    device: torch.device = None,
):
    device = device if device is not None else get_device()
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(all_preds, all_targets, class_counts)
    metrics["ECE"] = compute_ece(all_probs, all_targets)
    return metrics


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------
def method_list(args) -> list[tuple[str, str, str]]:
    """(label, kind, SelMix variant) of the methods to run, in table order."""
    methods = []
    if "MiSLAS" in args.methods:
        methods.append(("MiSLAS", "mislas", None))
    if "SelMix" in args.methods:
        for variant in args.selmix_variants:
            methods.append((f"MiSLAS+SelMix(10k) {variant}", "selmix", variant))
    if "LNR" in args.methods:
        methods.append(("MiSLAS+LNR", "lnr", None))
    return methods
METRIC_KEYS = ["Overall", "Many-shot", "Medium-shot", "Few-shot", "ECE"]
FLIP_KEYS = ["Flips (selected epoch)", "Flips (last epoch)", "Flips (mean per epoch)", "Flips (ever)"]
CSV_FIELDS = ["protocol", "seed", "method", "selected"] + METRIC_KEYS + FLIP_KEYS


def set_seed(seed: int):
    """Seeds Python's `random` (class-balanced sampler), NumPy (SelMix sampling) and PyTorch
    (augmentation, LNR Bernoulli draws; all devices) for one run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def protocol_tag(args) -> str:
    """Name of the evaluation protocol, used in the result file names."""
    return f"{args.epoch_selection}_{args.stage2_epochs}ep"


def run_folder_name(args, parser: argparse.ArgumentParser) -> str:
    """
    Automatic name of a run's result folder:
    cifar10_<protocol>_seeds<seeds>[_<option>-<value> for every option changed from its default]_<date>_<time>,
    e.g. cifar10_best_test_100ep_seeds0-4_t_flip-7.5_2026-10-30_14-05-12.
    """
    from datetime import datetime
    seeds = sorted(args.seeds)
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        seed_part = f"seeds{seeds[0]}-{seeds[-1]}"
    else:
        seed_part = "seeds" + "+".join(str(x) for x in args.seeds)
    parts = ["cifar10", protocol_tag(args), seed_part]
    # Options already in the name, or paths that do not change the experiment
    skip = {"seeds", "stage2_epochs", "epoch_selection", "out_dir", "summarize_only", "data_dir",
            "stage1_checkpoint"}
    for key, value in sorted(vars(args).items()):
        if key not in skip and value != parser.get_default(key):
            value = "+".join(str(v) for v in value) if isinstance(value, list) else value
            parts.append(f"{key}-{value}")
    parts.append(f"{datetime.now():%Y-%m-%d_%H-%M-%S}")
    return "_".join(parts)


def build_data(args, device: torch.device) -> dict:
    """Datasets, SelMix validation set and evaluation loader (identical for all seeds)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # rand_number must be the seed the stage-1 checkpoint was trained with (see --lt_seed)
    train_set = CIFAR10_LT(
        root=args.data_dir, train=True, ir=100.0, rand_number=args.lt_seed, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    # Same long-tailed training samples in the same order, without augmentation:
    # used by LNR to compute Q on the whole training set
    train_eval_set = CIFAR10_LT(
        root=args.data_dir, train=True, ir=100.0, rand_number=args.lt_seed, transform=test_transform
    )
    assert np.array_equal(train_eval_set.targets, train_set.targets)

    # SelMix's 10k validation set (option (b) in the report, see `build_selmix_val_set`):
    # 500 unused CIFAR-10 training images per class for classes 1-9 and 500 class-0 test images,
    # plus flipped copies. The class-0 test images are removed from the evaluation set of ALL
    # methods, so no method is trained or guided on an evaluation image.
    selmix_val_set, eval_idx = build_selmix_val_set(args.data_dir, train_set, test_set, test_transform)
    print(f"SelMix validation set: {len(selmix_val_set)} images; evaluation set: {len(eval_idx)} test images")

    # Pin memory only on CUDA to avoid MPS UserWarning
    test_loader = DataLoader(
        Subset(test_set, eval_idx), batch_size=100, shuffle=False, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    return {
        "train_set": train_set,
        "train_eval_set": train_eval_set,
        "selmix_val_set": selmix_val_set,
        "test_loader": test_loader,
        "class_counts": train_set.class_counts,
    }


def run_method(method: str, kind: str, variant: str, args, data: dict, device: torch.device) -> dict:
    """
    Runs one method once (with the current seed) and returns its result row.

    --epoch_selection last:      evaluate once, after the last Stage-2 epoch (SelMix: last step).
    --epoch_selection best_test: evaluate on the test set after every epoch (SelMix variants
                                 frozen and full: every --selmix_eval_every steps) and keep the
                                 epoch with the highest overall accuracy, as the LNR authors' code does.
    """
    class_counts = data["class_counts"]
    test_loader = data["test_loader"]
    select_best = args.epoch_selection == "best_test"

    model = ResNet32(num_classes=10).to(device)
    # Load the pre-trained Stage-1 checkpoint and freeze the backbone (fc + lws trainable)
    load_mislas_stage1_checkpoint(model, args.stage1_checkpoint, device)
    configure_mislas_stage2(model)

    if kind == "selmix":
        # SelMix replaces the MiSLAS Stage-2 loss with its own mixup loss; `variant` decides what
        # is trained and with which optimizer (see `selmix_finetune`)
        selmix = SelMix(data["train_set"], data["selmix_val_set"], device=device)
        # Steps of one MiSLAS Stage-2 epoch (class-balanced sampler: max class size x 10 samples)
        steps_per_epoch = math.ceil(max(class_counts) * len(class_counts) / args.batch_size)
        eval_fn = (lambda: evaluate(model, test_loader, class_counts, device=device)) if select_best else None
        result = selmix_finetune(
            model, selmix, args, device, eval_fn=eval_fn, eval_every=args.selmix_eval_every,
            variant=variant, steps_per_epoch=steps_per_epoch,
        )
        if result is None:
            metrics = evaluate(model, test_loader, class_counts, device=device)
            metrics["selected"] = (f"epoch {args.stage2_epochs}" if variant == "frozen-mislasopt"
                                   else f"step {args.selmix_steps}")
        else:
            (metrics, position), (last_metrics, last_position) = result
            metrics["selected"] = position
            metrics["_last"] = {**last_metrics, "selected": last_position}
        return metrics

    # Default MiSLAS CIFAR-10-LT (IF=100) Stage-2 hyperparameters
    # (Zhong et al., 2021, Table 6 and config/cifar10/cifar10_imb001_stage2_mislas.yaml)
    criterion = LabelAwareSmoothing(
        cls_num_list=class_counts, smooth_head=args.smooth_head, smooth_tail=args.smooth_tail,
        shape="concave",
    ).to(device)

    # As in MiSLAS: the classifier is trained with lr_factor x lr, the weight scaling with lr
    optimizer = torch.optim.SGD(
        [
            {"params": model.fc.parameters(), "lr": args.lr_factor * args.stage2_lr},
            {"params": model.lws.parameters(), "lr": args.stage2_lr},
        ],
        lr=args.stage2_lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    # Cosine schedule per epoch, lr_t = 0.5 * lr * (1 + cos(pi * t / T)), t = 0..T-1, per group
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage2_epochs, eta_min=0.0
    )
    # A fresh class-balanced sampler per run, so the sampling order depends only on the seed
    train_loader = DataLoader(
        data["train_set"], batch_size=args.batch_size,
        sampler=ClassBalancedSampler(data["train_set"].targets, num_classes=10),
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    flip_counts = []  # number of flipped samples in each epoch (MiSLAS+LNR only)
    ever_flipped = torch.zeros(len(data["train_set"]), dtype=torch.bool, device=device)
    best_metrics, best_epoch = None, None
    for epoch in range(1, args.stage2_epochs + 1):
        flipped_targets = None
        if method == "MiSLAS+LNR":
            # Algorithm 2 at round t: Q from the current model on the whole training set
            probs, base_targets = compute_train_posteriors(model, data["train_eval_set"], device)
            # Stage-1 checkpoint is already converged; deferred_epochs=0 activates flipping immediately
            flipped_targets = lnr_multiclass_flip(
                probs=probs,
                targets=base_targets,
                class_counts=class_counts,
                t_flip=args.t_flip,
                epoch=epoch,
                deferred_epochs=0,
            )
            is_flipped = flipped_targets != base_targets
            ever_flipped |= is_flipped
            flip_counts.append(int(is_flipped.sum()))
            flips_into = torch.bincount(flipped_targets[is_flipped], minlength=10).tolist()
            print(
                f"LNR epoch [{epoch:02d}/{args.stage2_epochs:02d}] | "
                f"Flipped samples: {flip_counts[-1]} | "
                f"Flipped into classes 0-9: {flips_into}"
            )

        loss = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            class_counts=class_counts,
            method=method,
            epoch=epoch,
            t_flip=args.t_flip,
            flipped_targets=flipped_targets,
            shift_bn=args.shift_bn,
            device=device,
        )
        lr_scheduler.step()

        if select_best:
            # As in the LNR authors' code: evaluate on the test set after every epoch, keep the best
            metrics = evaluate(model, test_loader, class_counts, device=device)
            if best_metrics is None or metrics["Overall"] > best_metrics["Overall"]:
                best_metrics, best_epoch = metrics, epoch
            last_metrics = metrics  # after the loop: the evaluation after the last epoch

        if epoch % 5 == 0 or epoch == args.stage2_epochs:
            print(
                f"Stage-2 Epoch [{epoch:02d}/{args.stage2_epochs:02d}] | "
                f"Loss: {loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                + (f" | best Overall so far: {best_metrics['Overall']:.2f}% (epoch {best_epoch})" if select_best else "")
            )

    if select_best:
        metrics = best_metrics
    else:
        metrics, best_epoch = evaluate(model, test_loader, class_counts, device=device), args.stage2_epochs
    metrics["selected"] = f"epoch {best_epoch}"
    if flip_counts:
        metrics["Flips (selected epoch)"] = flip_counts[best_epoch - 1]
        metrics["Flips (last epoch)"] = flip_counts[-1]
        metrics["Flips (mean per epoch)"] = float(np.mean(flip_counts))
        metrics["Flips (ever)"] = int(ever_flipped.sum())  # distinct samples flipped in any epoch
    if select_best:
        # The same run evaluated after its last epoch (gives the last-epoch results without a new run)
        metrics["_last"] = {**last_metrics, "selected": f"epoch {args.stage2_epochs}"}
        if flip_counts:
            metrics["_last"].update({k: v for k, v in metrics.items() if k.startswith("Flips")})
            metrics["_last"]["Flips (selected epoch)"] = flip_counts[-1]
    return metrics


def write_rows(path: Path, rows: list[dict]):
    """Writes the result rows of one run (one seed) to a CSV file."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def summarize(out_dir: Path, tag: str):
    """
    Aggregates all runs of one protocol (files <tag>_seed*.csv in out_dir): mean and sample
    standard deviation (ddof = 1) over the seeds of every metric, including ECE (each run's ECE
    is computed on that run's predictions, then averaged). Writes <tag>_summary.csv and prints
    the table.
    """
    import csv
    files = sorted(out_dir.glob(f"{tag}_seed*.csv"))
    if not files:
        print(f"No result files {tag}_seed*.csv in {out_dir}")
        return
    rows = []
    for fpath in files:
        with open(fpath, newline="") as f:
            rows += list(csv.DictReader(f))

    methods = list(dict.fromkeys(r["method"] for r in rows))  # order of appearance
    summary = []
    for method in methods:
        runs = [r for r in rows if r["method"] == method]
        if not runs:
            continue
        entry = {"method": method, "runs": len(runs), "seeds": " ".join(r["seed"] for r in runs),
                 "selected": " ".join(r["selected"].split()[-1] for r in runs)}
        for key in METRIC_KEYS + FLIP_KEYS:
            vals = np.array([float(r[key]) for r in runs if r.get(key, "") != ""])
            if len(vals):
                entry[f"{key} mean"] = vals.mean()
                entry[f"{key} std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary.append(entry)

    fields = ["method", "runs", "seeds", "selected"] + [f"{k} {s}" for k in METRIC_KEYS + FLIP_KEYS for s in ("mean", "std")]
    with open(out_dir / f"{tag}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: (f"{entry[k]:.4f}" if isinstance(entry.get(k), float) else entry.get(k, "")) for k in fields})

    def cell(entry, key):
        return f"{entry[key + ' mean']:.2f}±{entry[key + ' std']:.2f}" if key + " mean" in entry else "-"

    print("\n" + "=" * 132)
    print(f"CIFAR-10-LT (IR=100) SUMMARY, protocol {tag}: mean ± std over {max(e['runs'] for e in summary)} run(s)")
    print("=" * 132)
    print(f"{'Method':<36} | {'Many-shot':<12} | {'Med-shot':<12} | {'Few-shot':<12} | {'Overall':<12} | "
          f"{'ECE':<12} | {'Flips/epoch':<13} | selected")
    print("-" * 132)
    for entry in summary:
        print(f"{entry['method']:<36} | {cell(entry, 'Many-shot'):<12} | {cell(entry, 'Medium-shot'):<12} | "
              f"{cell(entry, 'Few-shot'):<12} | {cell(entry, 'Overall'):<12} | {cell(entry, 'ECE'):<12} | "
              f"{cell(entry, 'Flips (mean per epoch)'):<13} | {entry['selected']}")
    print(f"\n-> {out_dir / (tag + '_summary.csv')}")


def run_benchmark(args):
    device = get_device()
    print(f"Executing CIFAR-10-LT (IR=100) MiSLAS Stage-2 Benchmark on: {device.type.upper()}")
    print(f"Dataset directory: {args.data_dir}")
    out_dir, tag = Path(args.out_dir), protocol_tag(args)
    print(f"Protocol: {tag} (epoch selection: {args.epoch_selection}, {args.stage2_epochs} Stage-2 epochs); "
          f"seeds: {args.seeds}; results: {out_dir}")

    # All settings of this run, saved next to its results
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data = build_data(args, device)

    # In best_test mode the runs are evaluated after every epoch anyway, so the results after the
    # last epoch of the same runs are written too (protocol last_<epochs>ep, files and summary)
    last_tag = f"last_{args.stage2_epochs}ep" if args.epoch_selection == "best_test" else None

    for seed in args.seeds:
        print(f"\n#################### Run with seed {seed} ####################")
        set_seed(seed)
        rows, last_rows = [], []

        # Sanity check of the checkpoint loading: the stage-1 model before any Stage-2 training
        # (MiSLAS README, CIFAR-10-LT IF=100 Stage-1 (mixup): 72.8% top-1, 2.14% ECE)
        model = ResNet32(num_classes=10).to(device)
        load_mislas_stage1_checkpoint(model, args.stage1_checkpoint, device)
        metrics = evaluate(model, data["test_loader"], data["class_counts"], device=device)
        rows.append({"protocol": tag, "seed": seed, "method": "Stage-1 checkpoint", "selected": "-", **metrics})
        last_rows.append({**rows[-1], "protocol": last_tag})
        print(f"Stage-1 checkpoint -> Overall: {metrics['Overall']:.2f}% | ECE: {metrics['ECE']:.2f}%")

        for method, kind, variant in method_list(args):
            print(f"\n==================== Stage-2 Fine-Tuning: {method} (seed {seed}) ====================")
            if variant is not None:
                print(f"SelMix variant {variant}: {SELMIX_VARIANTS[variant]}")
            set_seed(seed)  # each method's run depends only on the seed, not on the methods before it
            metrics = run_method(method, kind, variant, args, data, device)
            last_metrics = metrics.pop("_last", None)
            rows.append({"protocol": tag, "seed": seed, "method": method, **metrics})
            if last_metrics is not None:
                last_rows.append({"protocol": last_tag, "seed": seed, "method": method, **last_metrics})
            print(
                f"Results for {method:<34} -> Overall: {metrics['Overall']:.2f}% | "
                f"Many: {metrics['Many-shot']:.2f}% | Med: {metrics['Medium-shot']:.2f}% | "
                f"Few: {metrics['Few-shot']:.2f}% | ECE: {metrics['ECE']:.2f}% | selected: {metrics['selected']}"
            )
            if "Flips (ever)" in metrics:
                print(
                    f"Flipped samples for {method:<14} -> selected epoch: {metrics['Flips (selected epoch)']} | "
                    f"last epoch: {metrics['Flips (last epoch)']} | "
                    f"mean per epoch: {metrics['Flips (mean per epoch)']:.1f} | "
                    f"ever flipped: {metrics['Flips (ever)']} (paper: 44 on CIFAR-10-LT)"
                )

        # One file per run, written as soon as the run is finished
        write_rows(out_dir / f"{tag}_seed{seed}.csv", rows)
        print(f"-> {out_dir / f'{tag}_seed{seed}.csv'}")
        if last_tag is not None:
            write_rows(out_dir / f"{last_tag}_seed{seed}.csv", last_rows)
            print(f"-> {out_dir / f'{last_tag}_seed{seed}.csv'} (same run, evaluated after the last epoch)")

    summarize(out_dir, tag)
    if last_tag is not None:
        summarize(out_dir, last_tag)


if __name__ == "__main__":
    default_ckpt = DEFAULT_STAGE1_CHECKPOINT

    parser = argparse.ArgumentParser(description="CIFAR-10-LT Imbalanced Benchmark (MiSLAS Stage-2)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(data_root / "cifar10"),
        help="Directory where CIFAR-10 (cifar-10-batches-py) is stored or downloaded to (default: data/cifar10)",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=str(default_ckpt),
        help="Path to the MiSLAS Stage-1 checkpoint (default: data/cifar10/cifar10_imb001_stage1_mixup10.pth.tar)",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=10,
        help="MiSLAS default Stage-2 fine-tuning epochs (default: 10)",
    )
    parser.add_argument(
        "--stage2_lr",
        type=float,
        default=0.1,
        help="Initial learning rate for MiSLAS Stage-2 cosine schedule",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    # MiSLAS defaults for CIFAR-10-LT IF=100 Stage 2 (Zhong et al., 2021, Table 6, and
    # config/cifar10/cifar10_imb001_stage2_mislas.yaml in the MiSLAS repository)
    parser.add_argument(
        "--lr_factor", type=float, default=0.5,
        help="Learning-rate ratio of the classifier (fc) to the weight scaling (default: 0.5)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=2e-4, help="SGD weight decay (default: 2e-4)"
    )
    parser.add_argument(
        "--smooth_head", type=float, default=0.3, help="Label-aware smoothing of the largest class (default: 0.3)"
    )
    parser.add_argument(
        "--smooth_tail", type=float, default=0.0, help="Label-aware smoothing of the smallest class (default: 0.0)"
    )
    parser.add_argument(
        "--shift_bn", action="store_true",
        help="Re-estimate the BN running statistics in Stage 2 (MiSLAS CIFAR config default: off)",
    )
    parser.add_argument(
        "--lt_seed", type=int, default=0,
        help="Seed of the CIFAR-10-LT subsampling; must match the stage-1 checkpoint "
             "(MiSLAS's released checkpoint: 0, MiSLAS's code; the LNR fork uses 4)",
    )
    parser.add_argument(
        "--t_flip", type=float, default=3.5, help="LNR flipping threshold per Appendix C.2"
    )
    # SelMix defaults for supervised CIFAR-10-LT, rho = 100 (Ramasubramanian et al., 2024, Table G.2;
    # weight decay from the official configs); batch size: --batch_size (128, as in Table G.2)
    parser.add_argument(
        "--selmix_lr", type=float, default=3e-3,
        help="SelMix learning rate of the classifier; the backbone uses 1/10 of it (default: 3e-3)",
    )
    parser.add_argument(
        "--selmix_weight_decay", type=float, default=5e-4, help="SelMix weight decay (default: 5e-4)"
    )
    parser.add_argument(
        "--selmix_steps", type=int, default=2000, help="SelMix fine-tuning SGD steps (default: 2000)"
    )
    parser.add_argument(
        "--selmix_eval_every", type=int, default=50,
        help="With --epoch_selection best_test: evaluate SelMix (variants frozen, full) every this many "
             "steps (default: 50, the test-evaluation interval of the official SelMix code); "
             "frozen-mislasopt is evaluated every epoch",
    )
    parser.add_argument(
        "--selmix_variants", nargs="+", choices=list(SELMIX_VARIANTS), default=["frozen"],
        help="SelMix variants to run, one result row each: frozen-mislasopt (backbone frozen, MiSLAS "
             "Stage-2 optimizer and --stage2_epochs), frozen (backbone frozen, SelMix's optimizer and "
             "--selmix_steps; default), full (SelMix's own default: backbone fine-tuned)",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=["MiSLAS", "SelMix", "LNR"], default=["MiSLAS", "SelMix", "LNR"],
        help="Methods to run (default: all three)",
    )
    # Repeated runs and evaluation protocol
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seeds of the runs; one result file per seed, then mean ± std over all seeds (default: 0)",
    )
    parser.add_argument(
        "--epoch_selection", choices=["last", "best_test"], default="last",
        help="last: evaluate after the last Stage-2 epoch (default). best_test: keep the epoch with the "
             "highest test accuracy, as in the LNR authors' code (selects on the test set); also writes "
             "the last-epoch results of the same runs (last_<epochs>ep files and summary)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory of the result files (default: a new, automatically named folder per run, "
             "results/cifar10_<protocol>_seeds<seeds>[_<changed options>]_<date>_<time>, so earlier "
             "results are never overwritten)",
    )
    parser.add_argument(
        "--summarize_only", action="store_true",
        help="Only aggregate the existing result files of this protocol in --out_dir (no training)",
    )
    args = parser.parse_args()

    if args.summarize_only:
        if args.out_dir is None:
            parser.error("--summarize_only needs --out_dir, e.g. results/cifar10_last_10ep_seeds0-4_2026-10-30_14-05-12")
        summarize(Path(args.out_dir), protocol_tag(args))
    else:
        if args.out_dir is None:
            # One automatically named folder per invocation (protocol, seeds, changed options, start time)
            args.out_dir = str(PROJECT_ROOT / "results" / run_folder_name(args, parser))
        run_benchmark(args)
