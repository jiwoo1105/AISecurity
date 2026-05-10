"""
베이스라인 분류기 학습 스크립트
- ResNet-50 / DenseNet-121
- CheXpert 폐렴 Binary Classification
- 평가: AUC, Clean Accuracy
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.classifier import PneumoniaClassifier
from data.dataset import get_dataloaders
from evaluation.metrics import compute_accuracy, compute_auc, compute_ece


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(train_loader.dataset)
    auc = compute_auc(all_probs, all_labels)

    return avg_loss, auc


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = (all_probs >= 0.5).long()

    avg_loss = total_loss / len(data_loader.dataset)
    acc = compute_accuracy(preds, all_labels)
    auc = compute_auc(all_probs, all_labels)
    ece = compute_ece(all_probs, all_labels)

    return avg_loss, acc, auc, ece


def train(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    # Device
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed
    torch.manual_seed(cfg.get("seed", 42))

    # Data
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    csv_path = os.path.join(data_cfg["data_dir"], "train.csv")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=csv_path,
        data_dir=data_cfg["data_dir"],
        image_size=data_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=cfg.get("seed", 42),
    )

    # Model
    model_name = cfg["model"]["name"]
    model = PneumoniaClassifier(
        name=model_name,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    print(f"Model: {model_name}")

    # Loss / Optimizer / Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 5)
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{train_cfg['epochs']} ---")

        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_ece = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val ECE: {val_ece:.4f}")

        # Best model 저장
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
            }, save_path)
            print(f"Best model saved! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test 평가
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)

    # Best model 로드
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_auc, test_ece = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test ECE:      {test_ece:.4f}")

    # 결과 저장
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    import csv
    results_path = os.path.join(results_dir, "baseline_eval.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "clean_accuracy", "auc", "ece"])
        writer.writerow([model_name, f"{test_acc:.4f}", f"{test_auc:.4f}", f"{test_ece:.4f}"])

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    train(config)
