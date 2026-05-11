"""
방어 실행 스크립트
- 공격된 이미지에 방어 적용 후 Robust Accuracy 평가
- Vanilla DiffPure, Randomized Smoothing 방어 비교
"""

import os
import sys
import csv
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.classifier import PneumoniaClassifier
from data.dataset import get_dataloaders
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.diff_attack import DiffAttack
from defense.vanilla_diffpure import VanillaDiffPure
from defense.randomized_smoothing import RandomizedSmoothing
from evaluation.metrics import compute_accuracy, compute_auc


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(cfg, device):
    model_name = cfg["model"]["name"]
    model = PneumoniaClassifier(name=model_name, pretrained=False).to(device)

    checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")

    model.eval()
    return model


def evaluate_defense(model, attack, defense, test_loader, device, defense_name, attack_name):
    """공격 → 방어 → 평가"""
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 공격
        adv_images = attack.attack(images, labels)

        # 방어
        if isinstance(defense, VanillaDiffPure):
            purified = defense.purify(adv_images)
            with torch.no_grad():
                logits = model(purified)
                preds = (torch.sigmoid(logits) >= 0.5).long()
        elif isinstance(defense, RandomizedSmoothing):
            preds, _ = defense.predict(adv_images, device)
        else:
            # 방어 없음
            with torch.no_grad():
                logits = model(adv_images)
                preds = (torch.sigmoid(logits) >= 0.5).long()

        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    robust_acc = correct / total
    print(f"  {defense_name} vs {attack_name}: Robust Acc = {robust_acc:.4f}")
    return robust_acc


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    device_name = cfg.get("device", "cpu")
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 모델 + 데이터
    model = load_model(cfg, device)
    data_cfg = cfg["data"]
    csv_path = os.path.join(data_cfg["data_dir"], "train.csv")
    _, _, test_loader = get_dataloaders(
        csv_path=csv_path, data_dir=data_cfg["data_dir"],
        image_size=data_cfg["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg["num_workers"], seed=cfg.get("seed", 42),
    )

    # 공격 정의
    eps = 0.04
    attacks = {
        "FGSM": FGSM(model, epsilon=eps),
        "PGD": PGD(model, epsilon=eps, step_size=0.01, num_steps=20),
        "DiffAttack": DiffAttack(model, epsilon=eps, num_steps=50),
    }

    # 방어 정의
    defenses = {
        "No Defense": None,
        "Vanilla DiffPure": VanillaDiffPure(noise_level=0.1),
        "Randomized Smoothing": RandomizedSmoothing(model, sigma=0.25, n_samples=50),
    }

    # 전체 평가
    results = []
    for attack_name, attack in attacks.items():
        print(f"\n=== Attack: {attack_name} (ε={eps}) ===")
        for defense_name, defense in defenses.items():
            robust_acc = evaluate_defense(
                model, attack, defense, test_loader, device,
                defense_name, attack_name,
            )
            results.append({
                "attack": attack_name,
                "defense": defense_name,
                "robust_acc": robust_acc,
            })

    # 결과 저장
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "defense_results.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "defense", "robust_accuracy"])
        for r in results:
            writer.writerow([r["attack"], r["defense"], f"{r['robust_acc']:.4f}"])

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config)
