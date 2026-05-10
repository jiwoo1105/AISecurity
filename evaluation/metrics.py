"""
평가 지표 모듈 (핵심 7개)

공격 평가: ASR, LPIPS
방어 평가: Clean Accuracy, Robust Accuracy, AUC
임상 신뢰도: ECE, Anatomical SSIM
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_asr(original_preds, attacked_preds):
    """
    Attack Success Rate (공격 성공률)
    원래 맞췄는데 공격 후 틀린 비율
    """
    correct_mask = (original_preds == 1)  # 원래 정답인 것들
    if correct_mask.sum() == 0:
        return 0.0
    flipped = (original_preds[correct_mask] != attacked_preds[correct_mask])
    return flipped.float().mean().item()


def compute_accuracy(preds, labels):
    """Clean / Robust Accuracy"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return accuracy_score(labels, preds)


def compute_auc(probs, labels):
    """AUC (Area Under ROC Curve)"""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.0


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error
    모델의 예측 확신도와 실제 정답률의 괴리 측정
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= low) & (probs < high)
        count = mask.sum()

        if count == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_anatomical_ssim(original, purified, segmenter):
    """
    Anatomical SSIM - 폐 영역만 비교하는 구조적 유사도

    Args:
        original: 원본 이미지 (B, C, H, W)
        purified: 정화된 이미지 (B, C, H, W)
        segmenter: 폐 분할 모델

    Returns:
        폐 영역 SSIM 값
    """
    from torchvision.transforms.functional import rgb_to_grayscale

    with torch.no_grad():
        mask = segmenter(original)
        mask = (mask > 0.5).float()

    # 폐 영역만 추출
    orig_lung = original * mask
    puri_lung = purified * mask

    # SSIM 계산 (폐 영역)
    return _ssim(orig_lung, puri_lung).item()


def _ssim(x, y, window_size=11):
    """SSIM 계산 (간략 버전)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.functional.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = torch.nn.functional.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = torch.nn.functional.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean()
