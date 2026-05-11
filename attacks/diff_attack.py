"""
DiffAttack - Stable Diffusion 기반 적대적 공격
- 실제 Diffusion 모델의 잠재 공간(latent space)에서 공격 수행
- 해부학적으로 자연스러운 변형 생성
- 중주파 영역에 공격 신호가 분포 (PGD의 고주파와 다름)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel


class DiffAttack:
    """Stable Diffusion 기반 적대적 공격"""

    def __init__(self, model, epsilon=0.04, num_steps=50, step_size=0.005,
                 diffusion_steps=20, criterion=None, device="cuda"):
        """
        Args:
            model: 타겟 분류기
            epsilon: perturbation 크기 제한
            num_steps: 공격 최적화 반복 횟수
            step_size: 잠재 공간에서의 step 크기
            diffusion_steps: diffusion forward/reverse step 수
            criterion: 손실 함수
            device: 연산 장치
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.diffusion_steps = diffusion_steps
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.device = device

        # Stable Diffusion VAE 로드 (이미지 ↔ 잠재공간 변환)
        print("Loading Stable Diffusion VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32,
        ).to(device)
        self.vae.eval()

        # Diffusion 노이즈 스케줄
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
        )

        print("DiffAttack initialized (Stable Diffusion VAE)")

    @torch.no_grad()
    def _encode(self, images):
        """이미지 → 잠재 공간 인코딩"""
        # VAE는 3채널 입력 필요, [-1, 1] 범위
        x = images * 2.0 - 1.0  # [0,1] → [-1,1]
        latent = self.vae.encode(x).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def _decode(self, latent):
        """잠재 공간 → 이미지 디코딩"""
        latent = latent / self.vae.config.scaling_factor
        decoded = self.vae.decode(latent).sample
        decoded = (decoded + 1.0) / 2.0  # [-1,1] → [0,1]
        return decoded.clamp(0, 1)

    def _add_noise(self, latent, timestep):
        """잠재 공간에 diffusion 노이즈 추가"""
        noise = torch.randn_like(latent)
        noisy = self.scheduler.add_noise(latent, noise, torch.tensor([timestep]))
        return noisy, noise

    def _denoise_step(self, noisy_latent, noise, timestep):
        """간략화된 denoise step"""
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        denoised = (noisy_latent - (1 - alpha_t).sqrt() * noise) / alpha_t.sqrt()
        return denoised

    def attack(self, images, labels):
        """
        DiffAttack 수행

        1. 이미지를 VAE로 잠재 공간에 인코딩
        2. 잠재 공간에서 분류기를 속이는 방향으로 최적화
        3. VAE로 다시 이미지 디코딩
        → 자연스러운 변형 (중주파 영역에 집중)

        Args:
            images: 원본 이미지 (B, C, H, W)
            labels: 정답 라벨 (B,)

        Returns:
            adv_images: 적대적 이미지 (B, C, H, W)
        """
        # Step 1: 이미지 → 잠재 공간
        latent = self._encode(images)

        # Step 2: 중간 timestep에서 노이즈 추가
        t = self.diffusion_steps
        noisy_latent, noise = self._add_noise(latent, t)

        # Step 3: 잠재 공간에서 적대적 perturbation 탐색
        delta = torch.zeros_like(latent, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # 변형된 잠재 표현
            perturbed_latent = noisy_latent.detach() + delta

            # denoise 후 이미지 복원
            denoised_latent = self._denoise_step(perturbed_latent, noise, t)

            # VAE 디코딩 (gradient 흐르게)
            denoised_latent_unscaled = denoised_latent / self.vae.config.scaling_factor
            decoded = self.vae.decode(denoised_latent_unscaled).sample
            adv_images = ((decoded + 1.0) / 2.0).clamp(0, 1)

            # 분류기 손실
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)

            # 역전파
            loss.backward()
            optimizer.step()

            # delta 크기 제한
            with torch.no_grad():
                delta.data.clamp_(-self.epsilon * 3, self.epsilon * 3)

        # Step 4: 최종 적대적 이미지 생성
        with torch.no_grad():
            final_latent = noisy_latent + delta
            denoised = self._denoise_step(final_latent, noise, t)
            adv_images = self._decode(denoised)

            # 원본과의 차이를 ε 범위로 제한
            perturbation = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + perturbation, 0, 1)

        return adv_images.detach()

    def attack_batch(self, dataloader, device):
        """
        전체 데이터에 대해 공격 수행

        Returns:
            dict: 원본/공격 이미지, 라벨, 예측 결과
        """
        self.model.eval()
        all_originals, all_advs, all_labels = [], [], []
        all_orig_preds, all_adv_preds = [], []

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # 공격 수행
            adv_images = self.attack(images, labels)

            # 예측
            with torch.no_grad():
                orig_logits = self.model(images)
                adv_logits = self.model(adv_images)
                orig_preds = (torch.sigmoid(orig_logits) >= 0.5).long()
                adv_preds = (torch.sigmoid(adv_logits) >= 0.5).long()

            all_originals.append(images.cpu())
            all_advs.append(adv_images.cpu())
            all_labels.append(labels.cpu())
            all_orig_preds.append(orig_preds.cpu())
            all_adv_preds.append(adv_preds.cpu())

            if (batch_idx + 1) % 5 == 0:
                print(f"  DiffAttack batch {batch_idx + 1} done")

        return {
            "originals": torch.cat(all_originals),
            "adversarials": torch.cat(all_advs),
            "labels": torch.cat(all_labels),
            "orig_preds": torch.cat(all_orig_preds),
            "adv_preds": torch.cat(all_adv_preds),
        }
