"""
공격 실행 스크립트
- FGSM / PGD 공격 수행
- 평가: ASR, LPIPS
- FFT 스펙트럼 시각화
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
from evaluation.metrics import compute_asr
from defense.fft_analyzer import FFTAnalyzer


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(cfg, device):
    """학습된 분류기 로드"""
    model_name = cfg["model"]["name"]
    model = PneumoniaClassifier(name=model_name, pretrained=False).to(device)

    checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

    model.eval()
    return model


def run_fgsm_attack(model, dataloader, epsilons, device):
    """다양한 ε에서 FGSM 공격 수행"""
    results = []

    for eps in epsilons:
        print(f"\n--- FGSM ε={eps} ---")
        attacker = FGSM(model, epsilon=eps)
        result = attacker.attack_batch(dataloader, device)

        asr = compute_asr(result["orig_preds"], result["adv_preds"])
        print(f"ASR: {asr:.4f}")

        results.append({
            "attack": "FGSM",
            "epsilon": eps,
            "asr": asr,
            "result": result,
        })

    return results


def run_pgd_attack(model, dataloader, epsilons, num_steps, step_size, device):
    """다양한 ε에서 PGD 공격 수행"""
    results = []

    for eps in epsilons:
        print(f"\n--- PGD ε={eps}, steps={num_steps} ---")
        attacker = PGD(model, epsilon=eps, step_size=step_size, num_steps=num_steps)
        result = attacker.attack_batch(dataloader, device)

        asr = compute_asr(result["orig_preds"], result["adv_preds"])
        print(f"ASR: {asr:.4f}")

        results.append({
            "attack": "PGD",
            "epsilon": eps,
            "asr": asr,
            "result": result,
        })

    return results


def save_results(all_results, output_path):
    """결과를 CSV로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "epsilon", "asr"])
        for r in all_results:
            writer.writerow([r["attack"], r["epsilon"], f"{r['asr']:.4f}"])

    print(f"\nResults saved to {output_path}")


def visualize_fft_comparison(results, save_dir):
    """원본 vs 공격 이미지 FFT 스펙트럼 비교 시각화"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    analyzer = FFTAnalyzer()

    for r in results:
        if r["epsilon"] != 0.04:  # 대표 ε만 시각화
            continue

        orig = r["result"]["originals"][0]   # 첫 번째 이미지
        adv = r["result"]["adversarials"][0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 원본
        axes[0, 0].imshow(orig[0].cpu().numpy(), cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        # 공격 이미지
        axes[0, 1].imshow(adv[0].cpu().numpy(), cmap="gray")
        axes[0, 1].set_title(f"{r['attack']} (ε={r['epsilon']})")
        axes[0, 1].axis("off")

        # 원본 FFT
        import torch.fft as fft_module
        orig_fft = torch.log1p(torch.abs(fft_module.fftshift(fft_module.fft2(orig[0]))))
        axes[1, 0].imshow(orig_fft.cpu().numpy(), cmap="hot")
        axes[1, 0].set_title("Original FFT")
        axes[1, 0].axis("off")

        # 공격 FFT
        adv_fft = torch.log1p(torch.abs(fft_module.fftshift(fft_module.fft2(adv[0]))))
        axes[1, 1].imshow(adv_fft.cpu().numpy(), cmap="hot")
        axes[1, 1].set_title(f"{r['attack']} FFT")
        axes[1, 1].axis("off")

        plt.suptitle(f"{r['attack']} Attack - FFT Comparison (ASR={r['asr']:.2%})")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"fft_{r['attack'].lower()}_eps{r['epsilon']}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"FFT visualization saved: {save_path}")


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    model = load_model(cfg, device)

    # 데이터 로드 (test set만 사용)
    data_cfg = cfg["data"]
    csv_path = os.path.join(data_cfg["data_dir"], "train.csv")
    _, _, test_loader = get_dataloaders(
        csv_path=csv_path,
        data_dir=data_cfg["data_dir"],
        image_size=data_cfg["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg.get("seed", 42),
    )

    # FGSM 공격
    attack_cfg = cfg["attack"]
    fgsm_results = run_fgsm_attack(
        model, test_loader, attack_cfg["fgsm"]["epsilons"], device
    )

    # PGD 공격
    pgd_results = run_pgd_attack(
        model, test_loader,
        attack_cfg["pgd"]["epsilons"],
        attack_cfg["pgd"]["steps"][1],  # 기본 20 steps
        attack_cfg["pgd"]["step_size"],
        device,
    )

    # 결과 저장
    all_results = fgsm_results + pgd_results
    results_dir = cfg["paths"]["results_dir"]
    save_results(all_results, os.path.join(results_dir, "attack_results.csv"))

    # FFT 시각화
    visualize_fft_comparison(all_results, os.path.join(results_dir, "fft_visualizations"))


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config)
