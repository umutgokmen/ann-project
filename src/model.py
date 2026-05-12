from pathlib import Path

import torch
import torch.nn as nn
import timm


class CarClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_feature_dim(self) -> int:
        return self.backbone.num_features


def build_model(cfg: dict) -> CarClassifier:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return CarClassifier(
        backbone=model_cfg["backbone"],
        num_classes=data_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.3),
        pretrained=model_cfg.get("pretrained", True),
    )


def load_checkpoint(model: CarClassifier, checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
