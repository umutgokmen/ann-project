import sys
import time
import math
import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import AverageMeter, accuracy, MixupCutmix


def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    train_cfg = cfg["training"]
    return optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg = cfg["training"]
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = train_cfg.get("warmup_epochs", 5) * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg, mixup_fn):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            soft_labels = True
        else:
            soft_labels = False

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)

        if not soft_labels:
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            acc_meter.update(top1, bs)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.2f}%")
        else:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(top1, bs)
        top5_meter.update(top5, bs)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    class_names = loaders["class_names"]

    # Save class mapping for inference
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['backbone']} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_cfg = cfg["training"]
    label_smoothing = train_cfg.get("label_smoothing", 0.1)

    mixup_fn = MixupCutmix(
        mixup_alpha=cfg["augmentation"].get("mixup_alpha", 0.2),
        cutmix_alpha=cfg["augmentation"].get("cutmix_alpha", 1.0),
        num_classes=cfg["data"]["num_classes"],
    )

    criterion_train = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")

    writer = SummaryWriter(log_dir=cfg["logging"]["log_dir"])

    best_top1 = 0.0
    save_top_k = cfg["logging"].get("save_top_k", 3)
    top_checkpoints: list[tuple[float, Path]] = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_train, optimizer, scheduler, scaler, device, cfg, mixup_fn
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion_eval, device, cfg)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{train_cfg['epochs']} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val_top1", val_top1, epoch)
        writer.add_scalar("Acc/val_top5", val_top5, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}_top1_{val_top1:.2f}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_top1": val_top1,
            "val_top5": val_top5,
            "cfg": cfg,
        }
        save_checkpoint(state, ckpt_path)
        top_checkpoints.append((val_top1, ckpt_path))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove checkpoints outside top-k
        while len(top_checkpoints) > save_top_k:
            _, old_path = top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        if val_top1 > best_top1:
            best_top1 = val_top1
            save_checkpoint(state, checkpoint_dir / "best.pt")
            print(f"  -> New best: {best_top1:.2f}%")

    writer.close()
    print(f"\nTraining complete. Best Val Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
