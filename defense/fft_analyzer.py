"""
FFT 주파수 분석 모듈
- 입력 X-ray를 저/중/고주파 대역으로 분리
- z-score 기반 공격 유형 탐지 (FGSM/PGD=고주파, DiffAttack=중주파)
- Attack Map 생성 (차등 노이즈 적용용)
"""

import torch
import torch.fft as fft
import numpy as np


class FFTAnalyzer:
    """FFT 기반 적대적 공격 탐지기"""

    def __init__(self, normal_stats=None, high_threshold=3.0, mid_threshold=2.5):
        """
        Args:
            normal_stats: 정상 X-ray 주파수 통계 {'low': {'mean':, 'std':}, 'mid':..., 'high':...}
            high_threshold: 고주파 이상 z-score 임계값
            mid_threshold: 중주파 이상 z-score 임계값
        """
        self.normal_stats = normal_stats
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold

    def compute_frequency_bands(self, image):
        """
        이미지를 저/중/고주파 대역으로 분리

        Args:
            image: 입력 이미지 (B, C, H, W) 또는 (C, H, W)

        Returns:
            dict: 각 대역의 magnitude
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 2D FFT → 주파수 영역 변환
        f = fft.fft2(image)
        f_shift = fft.fftshift(f)
        magnitude = torch.abs(f_shift)

        h, w = image.shape[-2:]
        cy, cx = h // 2, w // 2

        # 주파수 대역 마스크 생성
        y = torch.arange(h, device=image.device).float() - cy
        x = torch.arange(w, device=image.device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx**2 + yy**2)

        max_radius = min(cy, cx)
        r_low = max_radius / 3       # 저주파 경계
        r_mid = max_radius * 2 / 3   # 중주파 경계

        low_mask = (dist <= r_low).float()
        mid_mask = ((dist > r_low) & (dist <= r_mid)).float()
        high_mask = (dist > r_mid).float()

        return {
            "low": (magnitude * low_mask).mean(dim=(-2, -1)),
            "mid": (magnitude * mid_mask).mean(dim=(-2, -1)),
            "high": (magnitude * high_mask).mean(dim=(-2, -1)),
            "magnitude": magnitude,
            "low_mask": low_mask,
            "mid_mask": mid_mask,
            "high_mask": high_mask,
        }

    def detect_attack_type(self, image):
        """
        z-score 기반 공격 유형 탐지

        Returns:
            str: 'pixel_attack' (FGSM/PGD), 'diffusion_attack' (DiffAttack), 'clean'
        """
        if self.normal_stats is None:
            raise ValueError("정상 X-ray 통계가 필요합니다. compute_normal_stats()로 먼저 계산하세요.")

        bands = self.compute_frequency_bands(image)
        z_scores = {}

        for band_name in ["low", "mid", "high"]:
            energy = bands[band_name].mean().item()
            mean = self.normal_stats[band_name]["mean"]
            std = self.normal_stats[band_name]["std"]
            z_scores[band_name] = (energy - mean) / (std + 1e-8)

        if z_scores["high"] > self.high_threshold:
            return "pixel_attack", z_scores
        elif z_scores["mid"] > self.mid_threshold:
            return "diffusion_attack", z_scores
        else:
            return "clean", z_scores

    def generate_attack_map(self, image):
        """
        공격 영역 마스크 생성 (차등 노이즈 적용용)

        Returns:
            attack_map: (B, 1, H, W) 범위 [0, 1], 공격 의심 영역일수록 1에 가까움
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        bands = self.compute_frequency_bands(image)
        magnitude = bands["magnitude"]

        if self.normal_stats is None:
            # 통계 없으면 고주파 에너지 기반 단순 맵
            high_energy = magnitude * bands["high_mask"]
            attack_map = high_energy / (high_energy.max() + 1e-8)
        else:
            # z-score 기반 이상 영역 맵
            attack_type, z_scores = self.detect_attack_type(image)

            if attack_type == "pixel_attack":
                # 고주파 영역 강조
                target_energy = magnitude * bands["high_mask"]
            elif attack_type == "diffusion_attack":
                # 중주파 영역 강조
                target_energy = magnitude * bands["mid_mask"]
            else:
                # clean → 균일한 낮은 맵
                return torch.ones(image.shape[0], 1, image.shape[2], image.shape[3],
                                  device=image.device) * 0.1

            # IFFT로 공간 도메인 맵 생성
            f_shift = fft.fftshift(fft.fft2(image))
            filtered = f_shift * (bands["high_mask"] + bands["mid_mask"])
            spatial_energy = torch.abs(fft.ifft2(fft.ifftshift(filtered)))

            # [0, 1] 정규화
            attack_map = spatial_energy.mean(dim=1, keepdim=True)
            attack_map = (attack_map - attack_map.min()) / (attack_map.max() - attack_map.min() + 1e-8)

        return attack_map.mean(dim=1, keepdim=True)

    def visualize_spectrum(self, image, title="FFT Spectrum"):
        """
        주파수 스펙트럼 시각화 (matplotlib)

        Args:
            image: 입력 이미지 (C, H, W)
            title: 그래프 제목

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        if image.dim() == 4:
            image = image[0]

        # Grayscale 변환
        if image.shape[0] == 3:
            gray = image.mean(dim=0)
        else:
            gray = image[0]

        # FFT
        f = fft.fft2(gray)
        f_shift = fft.fftshift(f)
        magnitude = torch.log1p(torch.abs(f_shift)).cpu().numpy()

        # 대역별 에너지
        bands = self.compute_frequency_bands(image.unsqueeze(0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 원본 이미지
        axes[0].imshow(gray.cpu().numpy(), cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # FFT 스펙트럼
        axes[1].imshow(magnitude, cmap="hot")
        axes[1].set_title("FFT Magnitude Spectrum")
        axes[1].axis("off")

        # 대역별 에너지 바 차트
        band_names = ["Low", "Mid", "High"]
        energies = [
            bands["low"].mean().item(),
            bands["mid"].mean().item(),
            bands["high"].mean().item(),
        ]
        colors = ["#2196F3", "#FF9800", "#F44336"]
        axes[2].bar(band_names, energies, color=colors)
        axes[2].set_title("Frequency Band Energy")
        axes[2].set_ylabel("Mean Energy")

        fig.suptitle(title)
        plt.tight_layout()
        return fig


def compute_normal_stats(dataloader, device="cpu"):
    """
    정상 X-ray 데이터에서 주파수 대역 통계 계산

    Args:
        dataloader: 정상 이미지 DataLoader
        device: 연산 장치

    Returns:
        dict: {'low': {'mean': float, 'std': float}, 'mid': ..., 'high': ...}
    """
    analyzer = FFTAnalyzer()
    all_bands = {"low": [], "mid": [], "high": []}

    for images, _ in dataloader:
        images = images.to(device)
        bands = analyzer.compute_frequency_bands(images)

        for key in all_bands:
            all_bands[key].append(bands[key].mean(dim=-1).cpu())

    stats = {}
    for key in all_bands:
        values = torch.cat(all_bands[key]).numpy()
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return stats
