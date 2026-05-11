"""
Vanilla DiffPure - 일반 Diffusion 기반 적대적 정화 방어
- 공격된 이미지에 가우시안 노이즈를 추가한 후 Diffusion denoise로 정화
- 고주파 공격(FGSM/PGD)에 효과적
- 중주파 공격(DiffAttack)에는 한계 → Week 4 의료 특화 DiffPure로 개선

참고 논문: Nie et al., "Diffusion Models for Adversarial Purification" (ICML 2022)
"""

import torch
import torch.nn.functional as F


class VanillaDiffPure:
    """Vanilla DiffPure 방어"""

    def __init__(self, noise_level=0.1, denoise_steps=100):
        """
        Args:
            noise_level: 추가할 가우시안 노이즈의 강도 (0~1)
            denoise_steps: denoise 반복 횟수
        """
        self.noise_level = noise_level
        self.denoise_steps = denoise_steps

        # Diffusion 노이즈 스케줄
        self.betas = torch.linspace(1e-4, 0.02, denoise_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def purify(self, x_attacked):
        """
        적대적 이미지 정화

        과정:
        1. 가우시안 노이즈 추가 (공격 신호를 노이즈에 묻히게)
        2. 단계적 denoise (Diffusion reverse process 시뮬레이션)
        3. 정화된 이미지 반환

        Args:
            x_attacked: 공격받은 이미지 (B, C, H, W)

        Returns:
            x_purified: 정화된 이미지 (B, C, H, W)
        """
        device = x_attacked.device

        # Step 1: 노이즈 추가 (forward diffusion)
        t = int(self.denoise_steps * self.noise_level)
        t = max(1, min(t, self.denoise_steps - 1))

        alpha_t = self.alpha_cumprod[t].to(device)
        noise = torch.randn_like(x_attacked)
        x_noisy = torch.sqrt(alpha_t) * x_attacked + torch.sqrt(1 - alpha_t) * noise

        # Step 2: 단계적 denoise (reverse process)
        x_t = x_noisy
        for step in reversed(range(1, t + 1)):
            x_t = self._denoise_step(x_t, step)

        # Step 3: 클리핑
        x_purified = torch.clamp(x_t, 0, 1)

        return x_purified

    def _denoise_step(self, x_t, t):
        """
        단일 denoise step
        학습된 Diffusion 모델 대신 가우시안 스무딩 기반 근사 사용
        """
        alpha_t = self.alpha_cumprod[t].to(x_t.device)
        alpha_prev = self.alpha_cumprod[t - 1].to(x_t.device) if t > 0 else torch.tensor(1.0).to(x_t.device)
        beta_t = self.betas[t].to(x_t.device)

        # 노이즈 추정 (가우시안 블러로 근사)
        noise_level = (1 - alpha_t).sqrt().item()
        kernel_size = max(3, int(noise_level * 10) // 2 * 2 + 1)
        sigma = max(0.5, noise_level * 3)

        x_smooth = self._gaussian_blur(x_t, kernel_size, sigma)

        # Reverse step: 노이즈 제거 방향으로 이동
        noise_estimate = (x_t - torch.sqrt(alpha_t) * x_smooth) / (torch.sqrt(1 - alpha_t) + 1e-8)

        # DDPM reverse formula (간략화)
        x_prev = (1 / torch.sqrt(self.alphas[t].to(x_t.device))) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_t)) * noise_estimate
        )

        # 스토캐스틱 노이즈 추가 (t > 1일 때만)
        if t > 1:
            sigma_t = torch.sqrt(beta_t)
            x_prev = x_prev + sigma_t * torch.randn_like(x_t) * 0.3

        return x_prev

    def _gaussian_blur(self, x, kernel_size, sigma):
        """가우시안 블러 적용"""
        channels = x.shape[1]
        padding = kernel_size // 2

        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - padding
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)

    def purify_batch(self, dataloader, model, device):
        """
        전체 데이터에 대해 정화 + 평가

        Returns:
            dict: 정화 전/후 예측 결과
        """
        model.eval()
        all_purified_preds, all_labels = [], []
        all_orig_preds = []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 정화
            purified = self.purify(images)

            # 예측
            with torch.no_grad():
                orig_logits = model(images)
                puri_logits = model(purified)
                orig_preds = (torch.sigmoid(orig_logits) >= 0.5).long()
                puri_preds = (torch.sigmoid(puri_logits) >= 0.5).long()

            all_orig_preds.append(orig_preds.cpu())
            all_purified_preds.append(puri_preds.cpu())
            all_labels.append(labels.cpu())

        return {
            "orig_preds": torch.cat(all_orig_preds),
            "purified_preds": torch.cat(all_purified_preds),
            "labels": torch.cat(all_labels),
        }
