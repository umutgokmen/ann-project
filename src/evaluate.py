import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import build_dataloaders, get_transforms
from src.model import build_model, load_checkpoint
from src.utils import accuracy


@torch.no_grad()
def run_evaluation(model, loader, device, class_names, use_amp: bool = True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1 = (all_preds == all_labels).mean() * 100
    top5 = np.array([all_labels[i] in all_probs[i].argsort()[-5:] for i in range(len(all_labels))]).mean() * 100

    print(f"\nTest Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%")

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Per-class accuracy
    per_class_acc = defaultdict(list)
    for pred, label in zip(all_preds, all_labels):
        per_class_acc[label].append(pred == label)
    per_class_acc = {class_names[k]: np.mean(v) * 100 for k, v in per_class_acc.items()}

    worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:10]
    best_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop-10 Best Classes:")
    for cls, acc in best_classes:
        print(f"  {cls:<50} {acc:.1f}%")

    print("\nTop-10 Worst Classes:")
    for cls, acc in worst_classes:
        print(f"  {cls:<50} {acc:.1f}%")

    return all_preds, all_labels, all_probs, report


def plot_confusion_matrix(all_labels, all_preds, class_names, output_path: Path, top_n: int = 30):
    """Plot confusion matrix for the top_n most frequent classes."""
    class_counts = defaultdict(int)
    for label in all_labels:
        class_counts[label] += 1
    top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[:top_n]
    top_names = [class_names[i] for i in top_classes]

    mask = np.isin(all_labels, top_classes)
    filtered_labels = all_labels[mask]
    filtered_preds = all_preds[mask]

    label_map = {old: new for new, old in enumerate(top_classes)}
    mapped_labels = np.array([label_map[l] for l in filtered_labels])
    mapped_preds = np.array([label_map.get(p, -1) for p in filtered_preds])
    valid = mapped_preds != -1
    mapped_labels = mapped_labels[valid]
    mapped_preds = mapped_preds[valid]

    cm = confusion_matrix(mapped_labels, mapped_preds, labels=list(range(top_n)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=top_names, yticklabels=top_names)
    ax.set_title(f"Confusion Matrix (Top-{top_n} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to {output_path}")


def main(config_path: str, checkpoint_path: str, output_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    loaders = build_dataloaders(cfg)
    test_loader = loaders["test"]
    class_names = loaders["class_names"]

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} | Val Top-1: {ckpt.get('val_top1', '?'):.2f}%")

    all_preds, all_labels, all_probs, report = run_evaluation(
        model, test_loader, device, class_names,
        use_amp=cfg["training"].get("amp", True),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output_dir", default="logs/evaluation")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)
