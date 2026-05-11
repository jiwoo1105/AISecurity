"""
전체 분석 스크립트: DiffAttack + FFT 비교 + 방어 평가
- FGSM/PGD/DiffAttack 3종 공격 실행
- FFT 주파수 대역별 에너지 비교 (핵심 증명)
- Vanilla DiffPure / Randomized Smoothing 방어 평가
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.classifier import PneumoniaClassifier
from data.dataset import get_dataloaders
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.diff_attack import DiffAttack
from defense.vanilla_diffpure import VanillaDiffPure
from defense.randomized_smoothing import RandomizedSmoothing
from defense.fft_analyzer import FFTAnalyzer
from evaluation.metrics import compute_asr, compute_accuracy


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    model = PneumoniaClassifier(name="resnet50", pretrained=False).to(device)
    ckpt_path = "models/checkpoints/resnet50_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Model loaded (AUC: {ckpt.get('val_auc', 'N/A')})")
    model.eval()
    return model


def get_test_loader():
    """테스트 데이터 로드 (빠른 실행 위해 소량)"""
    _, _, test_loader = get_dataloaders(
        csv_path="data/chexpert/train.csv",
        data_dir="data/chexpert",
        image_size=224, batch_size=16, num_workers=0, seed=42,
    )
    return test_loader


def run_attacks(model, test_loader, device):
    """3종 공격 실행 + ASR 측정"""
    eps = 0.04
    attacks = {
        "FGSM": FGSM(model, epsilon=eps),
        "PGD": PGD(model, epsilon=eps, step_size=0.01, num_steps=20),
        "DiffAttack": DiffAttack(model, epsilon=eps, num_steps=30, diffusion_steps=30),
    }

    results = {}
    for name, attacker in attacks.items():
        print(f"\n--- {name} Attack (ε={eps}) ---")
        result = attacker.attack_batch(test_loader, device)
        asr = compute_asr(result["orig_preds"], result["adv_preds"])
        print(f"  ASR: {asr:.2%}")
        results[name] = {"result": result, "asr": asr}

    return results


def fft_comparison(attack_results, save_dir="results/fft_analysis"):
    """FFT 주파수 대역별 에너지 비교 (핵심 증명)"""
    os.makedirs(save_dir, exist_ok=True)
    analyzer = FFTAnalyzer()

    # 정상 통계 로드
    stats_path = "data/fft_statistics.pkl"
    normal_stats = None
    if os.path.exists(stats_path):
        normal_stats = torch.load(stats_path, weights_only=False)
        print(f"\n정상 FFT 통계 로드: {stats_path}")

    # 각 공격별 FFT 에너지 계산
    all_energies = {}
    sample_idx = 0

    for attack_name, data in attack_results.items():
        orig = data["result"]["originals"][sample_idx]
        adv = data["result"]["adversarials"][sample_idx]

        orig_bands = analyzer.compute_frequency_bands(orig.unsqueeze(0))
        adv_bands = analyzer.compute_frequency_bands(adv.unsqueeze(0))

        all_energies[attack_name] = {
            "orig": {k: orig_bands[k].mean().item() for k in ["low", "mid", "high"]},
            "adv": {k: adv_bands[k].mean().item() for k in ["low", "mid", "high"]},
        }

    # z-score 계산 (정상 통계가 있을 때)
    if normal_stats:
        print("\n=== z-score 분석 ===")
        for attack_name, energies in all_energies.items():
            print(f"\n{attack_name}:")
            for band in ["low", "mid", "high"]:
                z = (energies["adv"][band] - normal_stats[band]["mean"]) / (normal_stats[band]["std"] + 1e-8)
                print(f"  {band}: energy={energies['adv'][band]:.4f}, z-score={z:.2f}")

    # === 시각화 1: 대역별 에너지 비교 바 차트 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    band_names = ["Low", "Mid", "High"]
    x = range(3)
    width = 0.2

    # 원본
    orig_e = [all_energies["FGSM"]["orig"][k] for k in ["low", "mid", "high"]]
    ax.bar([i - 1.5*width for i in x], orig_e, width, label="Original", color="#2196F3")

    # 각 공격
    colors = {"FGSM": "#FF9800", "PGD": "#F44336", "DiffAttack": "#9C27B0"}
    for idx, (name, energies) in enumerate(all_energies.items()):
        adv_e = [energies["adv"][k] for k in ["low", "mid", "high"]]
        ax.bar([i + (idx - 0.5)*width for i in x], adv_e, width, label=name, color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=12)
    ax.set_ylabel("Mean Energy", fontsize=12)
    ax.set_title("FFT Frequency Band Energy: Original vs Attacks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fft_band_comparison.png"), dpi=150)
    plt.close()
    print(f"\n저장: {save_dir}/fft_band_comparison.png")

    # === 시각화 2: FFT 스펙트럼 비교 ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    import torch.fft as fft_module

    images_to_show = [("Original", attack_results["FGSM"]["result"]["originals"][sample_idx])]
    for name in ["FGSM", "PGD", "DiffAttack"]:
        images_to_show.append((name, attack_results[name]["result"]["adversarials"][sample_idx]))

    for i, (title, img) in enumerate(images_to_show):
        # 이미지
        axes[0, i].imshow(img[0].numpy(), cmap="gray")
        axes[0, i].set_title(title, fontsize=14)
        axes[0, i].axis("off")

        # FFT 스펙트럼
        f = fft_module.fft2(img[0])
        magnitude = torch.log1p(torch.abs(fft_module.fftshift(f)))
        axes[1, i].imshow(magnitude.numpy(), cmap="hot")
        axes[1, i].set_title(f"{title} FFT", fontsize=14)
        axes[1, i].axis("off")

    plt.suptitle("Image & FFT Spectrum Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fft_spectrum_comparison.png"), dpi=150)
    plt.close()
    print(f"저장: {save_dir}/fft_spectrum_comparison.png")

    # === 시각화 3: 에너지 변화량 (공격 - 원본) ===
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, energies in all_energies.items():
        diff = [energies["adv"][k] - energies["orig"][k] for k in ["low", "mid", "high"]]
        ax.bar([f"{b}\n({name})" for b in band_names], diff, color=colors[name], alpha=0.8)

    ax.set_ylabel("Energy Change (Attack - Original)", fontsize=12)
    ax.set_title("FFT Energy Change by Attack Type", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fft_energy_change.png"), dpi=150)
    plt.close()
    print(f"저장: {save_dir}/fft_energy_change.png")

    return all_energies


def run_defense(model, attack_results, test_loader, device):
    """방어 평가"""
    print("\n" + "=" * 60)
    print("방어 평가")
    print("=" * 60)

    diffpure = VanillaDiffPure(noise_level=0.1)
    rand_smooth = RandomizedSmoothing(model, sigma=0.25, n_samples=30)

    defense_results = []

    for attack_name, data in attack_results.items():
        adv_images = data["result"]["adversarials"]
        labels = data["result"]["labels"]

        # 방어 없음
        with torch.no_grad():
            logits = model(adv_images.to(device))
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
        no_def_acc = compute_accuracy(preds, labels)

        # Vanilla DiffPure
        purified = diffpure.purify(adv_images.to(device))
        with torch.no_grad():
            logits = model(purified)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
        diffpure_acc = compute_accuracy(preds, labels)

        # Randomized Smoothing
        rs_preds, _ = rand_smooth.predict(adv_images.to(device), device)
        rs_acc = compute_accuracy(rs_preds.cpu(), labels)

        print(f"\n{attack_name}:")
        print(f"  No Defense:           {no_def_acc:.2%}")
        print(f"  Vanilla DiffPure:     {diffpure_acc:.2%}")
        print(f"  Randomized Smoothing: {rs_acc:.2%}")

        defense_results.append({"attack": attack_name, "no_defense": no_def_acc,
                                "diffpure": diffpure_acc, "rand_smooth": rs_acc})

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    with open("results/defense_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "no_defense", "vanilla_diffpure", "randomized_smoothing"])
        for r in defense_results:
            writer.writerow([r["attack"], f"{r['no_defense']:.4f}",
                           f"{r['diffpure']:.4f}", f"{r['rand_smooth']:.4f}"])
    print(f"\n저장: results/defense_results.csv")

    # 방어 결과 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    attacks = [r["attack"] for r in defense_results]
    x = range(len(attacks))
    width = 0.25

    ax.bar([i - width for i in x], [r["no_defense"] for r in defense_results],
           width, label="No Defense", color="#F44336")
    ax.bar(x, [r["diffpure"] for r in defense_results],
           width, label="Vanilla DiffPure", color="#4CAF50")
    ax.bar([i + width for i in x], [r["rand_smooth"] for r in defense_results],
           width, label="Randomized Smoothing", color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=12)
    ax.set_ylabel("Robust Accuracy", fontsize=12)
    ax.set_title("Defense Performance vs Attacks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/defense_comparison.png", dpi=150)
    plt.close()
    print(f"저장: results/defense_comparison.png")

    return defense_results


def main():
    device = get_device()
    print(f"Device: {device}")

    model = load_model(device)
    test_loader = get_test_loader()

    # 1. 3종 공격 실행
    attack_results = run_attacks(model, test_loader, device)

    # 2. FFT 비교 분석
    fft_comparison(attack_results)

    # 3. 방어 평가
    run_defense(model, attack_results, test_loader, device)

    print("\n" + "=" * 60)
    print("전체 분석 완료!")
    print("=" * 60)
    print("결과 파일:")
    print("  - results/fft_analysis/fft_band_comparison.png")
    print("  - results/fft_analysis/fft_spectrum_comparison.png")
    print("  - results/fft_analysis/fft_energy_change.png")
    print("  - results/defense_results.csv")
    print("  - results/defense_comparison.png")


if __name__ == "__main__":
    main()
