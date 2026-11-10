"""
Utility modules for Long-Tailed CIFAR-10 (CIFAR-10-LT) experiments based on 
Hu et al. (ICML 2025), "Learning Imbalanced Data with Beneficial Label Noise".

Codes that are implemented from the following papers or adapted from their existing implementations:
    1. Long-Tailed CIFAR-10 construction (`get_cifar10_lt_counts`, `CIFAR10_LT`):
       Adapted from Cao et al. (NeurIPS 2019), "Learning Imbalanced Datasets with
       Label-Distribution-Aware Margin Loss", official repository:
       https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    2. CIFAR ResNet-32 architecture (`LambdaLayer`, `BasicBlock`, `ResNet32`):
       Adapted from He et al. (CVPR 2016) and Yerlan Idelbayev's CIFAR ResNet
       implementation used by LDAM-DRW and MiSLAS:
       https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
       https://github.com/JIA-Lab-research/MiSLAS/blob/main/models/resnet_cifar.py
    3. MiSLAS Stage-2 modules (`LearnableWeightScaling`, `LabelAwareSmoothing`,
       `ClassBalancedSampler`):
       Adapted from Zhong et al. (CVPR 2021), "Improving Calibration for Long-Tailed
       Recognition", official repository:
       https://github.com/JIA-Lab-research/MiSLAS/blob/main/mislas.py
       https://github.com/JIA-Lab-research/MiSLAS/blob/main/datasets/sampler.py
    4. Multi-Class LNR Algorithm 2 (`lnr_multiclass_flip`):
       Implemented from Hu et al. (ICML 2025), "Learning Imbalanced Data with
       Beneficial Label Noise", Algorithm 2, official repository:
       https://github.com/guangzhengh/LNR
"""

import math
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets


def get_device() -> torch.device:
    """Detects available hardware accelerator: CUDA, Apple Silicon MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Long-Tailed CIFAR-10 Dataset Construction & Class-Balanced Sampler
# ---------------------------------------------------------------------------
def get_cifar10_lt_counts(n_classes: int = 10, ir: float = 100.0, n_max: int = 5000) -> list[int]:
    """
    Computes sample sizes per class using exponential decay:
        N_i = floor(N_max * (1 / IR) ** (i / (C - 1))).
    """
    counts = []
    for i in range(n_classes):
        val = int(math.floor(n_max * ((1.0 / ir) ** (i / (n_classes - 1.0)))))
        counts.append(val)
    return counts


class CIFAR10_LT(Dataset):
    """
    Subsamples standard CIFAR-10 to create a long-tailed training set.
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            ir: float = 100.0,
            rand_number: int = 0,
            transform=None,
        ):
            self.transform = transform
            base_dataset = datasets.CIFAR10(root=root, train=train, download=True)
            self.data = base_dataset.data
            self.targets = np.array(base_dataset.targets, dtype=np.int64)
            self.classes = list(range(10))

            if train:
                # Save current numpy RNG state, apply rand_number=0 to match the exact
                # official MiSLAS / LDAM-DRW training split, then restore RNG state
                rng_state = np.random.get_state()
                np.random.seed(rand_number)

                self.class_counts = get_cifar10_lt_counts(n_classes=10, ir=ir, n_max=5000)
                selected_indices = []
                for cls_idx, count in enumerate(self.class_counts):
                    cls_indices = np.where(self.targets == cls_idx)[0]
                    np.random.shuffle(cls_indices)
                    selected_indices.extend(cls_indices[:count])

                np.random.set_state(rng_state)

                selected_indices = np.array(selected_indices)
                self.data = self.data[selected_indices]
                self.targets = self.targets[selected_indices]
                # Indices into the original CIFAR-10 training split; the images not listed
                # here are unused by CIFAR-10-LT (used for the SelMix validation set)
                self.indices = selected_indices
            else:
                self.class_counts = [1000] * 10  # Balanced evaluation set
                self.indices = np.arange(len(self.targets))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = int(self.targets[idx])
        # Convert numpy.ndarray (H, W, C) to PIL Image for torchvision transforms
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target, idx


class RandomCycleIter:
    """
    Endless iterator over `data` that reshuffles after every full pass.

    Source:
        Copied from `RandomCycleIter` in Zhong et al. (CVPR 2021),
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/datasets/sampler.py
    """

    def __init__(self, data, test_mode: bool = False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.i]


class ClassBalancedSampler(Sampler):
    """
    Class-balanced sampler for MiSLAS Stage-2 classifier fine-tuning.

    As in MiSLAS: classes are visited in a reshuffled cycle, `num_samples_cls` consecutive
    samples are drawn from each visited class (also cycling without replacement within the
    class), and one epoch has max(class size) x num_classes samples (5000 x 10 on CIFAR-10-LT).

    Adapted from `ClassAwareSampler` and `class_aware_sample_generator` in:
    - Kang et al. (ICLR 2020), "Decoupling Representation and Classifier for Long-Tailed Recognition"
    - Zhong et al. (CVPR 2021), https://github.com/JIA-Lab-research/MiSLAS/blob/main/datasets/sampler.py
    """

    def __init__(self, targets: np.ndarray, num_classes: int = 10, num_samples_cls: int = 4):
        super().__init__()
        self.targets = np.array(targets)
        self.num_classes = num_classes
        cls_indices = [np.where(self.targets == c)[0].tolist() for c in range(num_classes)]
        self.class_iter = RandomCycleIter(range(num_classes))
        self.data_iter_list = [RandomCycleIter(idx) for idx in cls_indices]
        self.num_samples = max(len(idx) for idx in cls_indices) * num_classes
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        i = 0
        while i < self.num_samples:
            data_iter = self.data_iter_list[next(self.class_iter)]
            for _ in range(self.num_samples_cls):
                if i >= self.num_samples:
                    break
                yield int(next(data_iter))
                i += 1

    def __len__(self):
        return self.num_samples


# ---------------------------------------------------------------------------
# 2. ResNet-32 Architecture & MiSLAS Learnable Weight Scaling (LWS)
# ---------------------------------------------------------------------------
class LambdaLayer(nn.Module):
    """
    Identity shortcut padding layer for CIFAR ResNet.

    Source:
        Copied from Yerlan Idelbayev's CIFAR ResNet implementation:
        - https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
        - Used in https://github.com/JIA-Lab-research/MiSLAS/blob/main/models/resnet_cifar.py
    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    Standard CIFAR ResNet BasicBlock with Option-A shortcut.

    Source / Attribution:
        Adapted from Yerlan Idelbayev's CIFAR ResNet implementation:
        - He et al. (CVPR 2016), "Deep Residual Learning for Image Recognition"
        - https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
        - Also used in https://github.com/JIA-Lab-research/MiSLAS/blob/main/models/resnet_cifar.py
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class LearnableWeightScaling(nn.Module):
    """
    MiSLAS Stage-2 Learnable Weight Scaling (LWS) module.

    Source:
        Copied from Zhong et al. (CVPR 2021), "Improving Calibration for Long-Tailed Recognition":
        - https://github.com/JIA-Lab-research/MiSLAS/blob/main/mislas.py
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.learned_norm * x


class ResNet32(nn.Module):
    """
    ResNet-32 backbone for CIFAR-10/100 long-tailed recognition.

    Source / Attribution:
        Adapted from:
        - https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
        - https://github.com/JIA-Lab-research/MiSLAS/blob/main/models/resnet_cifar.py
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.lws = LearnableWeightScaling(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = F.avg_pool2d(out, out.size()[3]).view(out.size(0), -1)
        logits = self.lws(self.fc(feat))
        return logits, feat


# ---------------------------------------------------------------------------
# 3. MiSLAS Label-Aware Smoothing (Zhong et al., CVPR 2021)
# ---------------------------------------------------------------------------
class LabelAwareSmoothing(nn.Module):
    """
    Label-Aware Smoothing (LAS) loss for MiSLAS Stage-2 classifier fine-tuning.

    Source:
        Adapted from Zhong et al. (CVPR 2021), "Improving Calibration for Long-Tailed Recognition":
        - https://github.com/JIA-Lab-research/MiSLAS/blob/main/mislas.py
        (Modified `self.smooth` registration to `register_buffer` for cross-device CUDA/MPS compatibility.)
    """

    def __init__(self, cls_num_list: list[int], smooth_head: float = 0.4, smooth_tail: float = 0.0, shape: str = "concave", power: float = None):
        super().__init__()
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        cls_num_arr = np.array(cls_num_list, dtype=np.float32)

        if shape == "concave":
            smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin(
                (cls_num_arr - n_K) * np.pi / (2 * (n_1 - n_K))
            )
        elif shape == "linear":
            smooth = smooth_tail + (smooth_head - smooth_tail) * (cls_num_arr - n_K) / (n_1 - n_K)
        elif shape == "convex":
            smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(
                1.5 * np.pi + (cls_num_arr - n_K) * np.pi / (2 * (n_1 - n_K))
            )
        elif shape == "exp" and power is not None:
            smooth = smooth_tail + (smooth_head - smooth_tail) * np.power(
                (cls_num_arr - n_K) / (n_1 - n_K), power
            )
        else:
            raise ValueError(f"Unsupported LAS shape: {shape}")

        self.register_buffer("smooth", torch.from_numpy(smooth).float())

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smoothing = self.smooth[target]
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# 4. Algorithm 2: Dynamic Multi-Class Label-Noise Rebalancing (LNR)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_train_posteriors(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device = None,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Q = softmax(fc(w_t; X, Y)) of the current model for every training sample
    (Algorithm 2 in Hu et al. (ICML 2025)), in dataset order, so row i belongs to the sample with index i.

    The model is put in evaluation mode and use the test transform for data
    (no augmentation).
    Returns (probs of shape (N, C), targets of shape (N,)), both on `device` (default: `get_device()`).
    """
    device = device if device is not None else get_device()
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=(device.type == "cuda"))
    all_probs, all_targets = [], []
    for inputs, targets, _ in loader:
        logits, _ = model(inputs.to(device))
        all_probs.append(F.softmax(logits, dim=1))
        all_targets.append(targets.to(device))
    model.train(was_training)
    return torch.cat(all_probs), torch.cat(all_targets)


def lnr_multiclass_flip(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_counts: list[int],
    t_flip: float = 3.5,
    epoch: int = 1,
    deferred_epochs: int = 80,
) -> torch.Tensor:
    """
    Implements Algorithm 2 in Hu et al. (ICML 2025): 
    Constructing label noises during training at round t.

    `probs` (N, C) and `targets` (N,) cover the whole training set, as returned by
    `compute_train_posteriors`. mu and nu are computed per class (column-wise) over all
    training samples.
    """
    if epoch <= deferred_epochs:
        return targets.clone()

    device = probs.device

    # 1. Posterior probabilities Q = softmax(fc(w_t; X, Y)) on the whole training set
    Q = probs.detach()  # Shape: (N, C)

    # 2. theta_C = 1 - minMax(N_C)
    counts = np.array(class_counts, dtype=np.float32)
    min_c, max_c = counts.min(), counts.max()
    minmax_norm = (counts - min_c) / (max_c - min_c + 1e-8)
    theta_C = torch.tensor(1.0 - minmax_norm, dtype=torch.float32, device=device)  # (C,)

    # 3. Per-class mu and nu over all training samples (one value per column of Q)
    mu = torch.mean(Q, dim=0, keepdim=True)  # (1, C)
    nu = torch.std(Q, dim=0, unbiased=False, keepdim=True) + 1e-8  # (1, C), ddof = 0

    # 4. Standardized Z-scores per sample and class: Z[x, k] = (Q[x, k] - mu_k) / nu_k
    Z = (Q - mu) / nu  # (N, C)

    # 5. Flip rates: F_x = max(tanh(Z_x - t_flip), 0) * max(theta_C - theta_C[y], 0)
    tanh_term = torch.clamp(torch.tanh(Z - t_flip), min=0.0)  # (N, C)
    theta_y = theta_C[targets].unsqueeze(1)  # (N, 1)
    strength = torch.clamp(theta_C.unsqueeze(0) - theta_y, min=0.0)  # (N, C)
    flip_rates = tanh_term * strength  # (N, C)

    # 6. Sample U ~ Bernoulli(F_x) and flip to highest triggered class (with random tie-breaking)
    rand_vals = torch.rand_like(flip_rates)
    triggered = (rand_vals < flip_rates) & (strength > 0)

    new_targets = targets.clone()
    has_triggers = triggered.any(dim=1)
    if has_triggers.any():
        for i in torch.where(has_triggers)[0]:
            valid_classes = torch.where(triggered[i])[0]
            if len(valid_classes) > 0:
                rates = flip_rates[i, valid_classes]
                max_rate = torch.max(rates)
                # Random tie-breaking per Section 4.3 of Hu et al. (ICML 2025)
                ties = valid_classes[torch.isclose(rates, max_rate)]
                chosen_idx = torch.randint(0, len(ties), (1,), device=device).item()
                new_targets[i] = ties[chosen_idx]

    return new_targets


# ---------------------------------------------------------------------------
# 5. Evaluation Helpers: Many / Medium / Few-shot and ECE
# ---------------------------------------------------------------------------
def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    class_counts: list[int],
    head_class_idx: tuple[int, int] = (0, 3),
    med_class_idx: tuple[int, int] = (3, 7),
    tail_class_idx: tuple[int, int] = (7, 10),
) -> dict[str, float]:
    """
    Computes Many-shot, Medium-shot, Few-shot, and Overall accuracy per Table 2
    and Appendix C.7 of Hu et al. (ICML 2025).

    The groups are class-index ranges [start, end), as in MiSLAS. The defaults are MiSLAS's
    CIFAR-10-LT (IF = 100) groups, with classes sorted by size
    N = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]:
    Many-shot = classes 0-2, Medium-shot = classes 3-6, Few-shot = classes 7-9.

    Source / Attribution:
        `head_class_idx`, `med_class_idx`, `tail_class_idx` in Zhong et al. (CVPR 2021),
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/config/cifar10/cifar10_imb001_stage2_mislas.yaml
        and their use in `validate` of
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/train_stage2.py
    """
    # Accuracies are means of per-class accuracies (as MiSLAS computes its group accuracies), so
    # every class counts equally even if the evaluation set has fewer images of some class
    # (class 0 has 500 test images left after the SelMix validation split). On a balanced
    # test set this equals the plain accuracy.
    per_class_acc = np.array([(preds[targets == c] == c).mean() for c in range(len(class_counts))]) * 100.0
    total_acc = per_class_acc.mean()

    many_acc = per_class_acc[head_class_idx[0]:head_class_idx[1]].mean()
    med_acc = per_class_acc[med_class_idx[0]:med_class_idx[1]].mean()
    few_acc = per_class_acc[tail_class_idx[0]:tail_class_idx[1]].mean()

    return {
        "Overall": total_acc,
        "Many-shot": many_acc,
        "Medium-shot": med_acc,
        "Few-shot": few_acc,
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) expressed as a percentage.

    Source:
        Adapted from Guo et al. (ICML 2017), "On Calibration of Modern Neural Networks",
        and `calibration` in Zhong et al. (CVPR 2021),
        https://github.com/JIA-Lab-research/MiSLAS/blob/main/utils/metric.py
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece * 100.0)