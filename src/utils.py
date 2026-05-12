import random
import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


class MixupCutmix:
    """Apply Mixup or CutMix randomly during training."""

    def __init__(self, mixup_alpha: float, cutmix_alpha: float, num_classes: int):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        if random.random() < 0.5:
            return self._mixup(images, labels)
        return self._cutmix(images, labels)

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        idx = torch.randperm(batch_size, device=images.device)

        mixed = lam * images + (1 - lam) * images[idx]
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[idx]
        return mixed, mixed_labels

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size, _, h, w = images.size()
        idx = torch.randperm(batch_size, device=images.device)

        cut_ratio = (1 - lam) ** 0.5
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cx = random.randint(0, w)
        cy = random.randint(0, h)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_actual = 1 - (y2 - y1) * (x2 - x1) / (h * w)

        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam_actual * labels_one_hot + (1 - lam_actual) * labels_one_hot[idx]
        return mixed, mixed_labels
