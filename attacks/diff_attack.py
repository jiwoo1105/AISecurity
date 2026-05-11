"""
DiffAttack - Diffusion 기반 적대적 공격
- Diffusion 모델의 잠재 공간(latent space)에서 공격 수행
- 해부학적으로 자연스러운 변형 생성 → 사람 눈에 구별 불가
- 중주파 영역에 공격 신호가 분포 (PGD의 고주파와 다름)

참고: 실제 Stable Diffusion 없이도 동작하는 Diffusion-style 공격 구현
      (Diffusion의 forward/reverse process를 시뮬레이션)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffAttack:
    """Diffusion 기반 적대적 공격"""

    def __init__(self, model, epsilon=0.04, num_steps=100, diffusion_steps=50,
                 step_size=0.002, criterion=None):
        """
        Args:
            model: 타겟 분류기
            epsilon: perturbation 크기 제한
            num_steps: 공격 최적화 반복 횟수
            diffusion_steps: diffusion forward/reverse step 수
            step_size: 각 step의 이동 크기
            criterion: 손실 함수
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.diffusion_steps = diffusion_steps
        self.step_size = step_size
        self.criterion = criterion or nn.BCEWithLogitsLoss()

        # Diffusion 노이즈 스케줄 (linear beta schedule)
        self.betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def _diffusion_forward(self, x, t):
        """
        Diffusion forward process: 이미지에 점진적 노이즈 추가

        Args:
            x: 원본 이미지 (B, C, H, W)
            t: timestep (0 ~ diffusion_steps-1)

        Returns:
            x_t: 노이즈가 추가된 이미지
            noise: 추가된 노이즈
        """
        alpha_t = self.alpha_cumprod[t].to(x.device)
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise

    def _diffusion_reverse(self, x_t, t):
        """
        Diffusion reverse process: 노이즈 제거 (간략화된 denoise)
        학습된 Diffusion 모델 대신 가우시안 스무딩으로 근사

        Args:
            x_t: 노이즈 이미지
            t: timestep

        Returns:
            x_denoised: 노이즈 제거된 이미지
        """
        alpha_t = self.alpha_cumprod[t].to(x_t.device)

        # 간략화된 denoise: 가우시안 블러 + alpha 보정
        kernel_size = max(3, int(5 * (1 - alpha_t.item())) // 2 * 2 + 1)
        sigma = max(0.5, 2.0 * (1 - alpha_t.item()))

        # 가우시안 커널 생성
        x_smooth = self._gaussian_blur(x_t, kernel_size, sigma)

        # alpha 보정으로 원본 스케일 복원
        x_denoised = (x_smooth - torch.sqrt(1 - alpha_t) * 0.1) / (torch.sqrt(alpha_t) + 1e-8)
        x_denoised = torch.clamp(x_denoised, 0, 1)

        return x_denoised

    def _gaussian_blur(self, x, kernel_size, sigma):
        """가우시안 블러 적용"""
        channels = x.shape[1]
        padding = kernel_size // 2

        # 1D 가우시안 커널 생성
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - padding
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 2D 커널로 확장
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)

    def attack(self, images, labels):
        """
        DiffAttack 수행

        1. 이미지를 diffusion forward로 잠재 공간에 매핑
        2. 잠재 공간에서 분류기를 속이는 방향으로 최적화
        3. diffusion reverse로 다시 이미지 복원
        → 자연스러운 변형이 만들어짐

        Args:
            images: 원본 이미지 (B, C, H, W)
            labels: 정답 라벨 (B,)

        Returns:
            adv_images: 적대적 이미지 (B, C, H, W)
        """
        # Step 1: Diffusion forward - 중간 timestep으로 노이즈 추가
        t = self.diffusion_steps // 3  # 전체의 1/3 지점 (너무 많이 노이즈 안 줌)
        x_t, noise = self._diffusion_forward(images, t)

        # Step 2: 잠재 공간에서 적대적 방향 탐색
        delta = torch.zeros_like(noise, requires_grad=True)

        for step in range(self.num_steps):
            # 변형된 잠재 표현
            x_t_adv = x_t + delta

            # Reverse process로 이미지 복원
            x_adv = self._diffusion_reverse(x_t_adv, t)

            # 분류기 손실 계산
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, labels)

            # Gradient 계산
            loss.backward()

            with torch.no_grad():
                # 잠재 공간에서 gradient 방향으로 이동
                grad = delta.grad.data

                # 중주파 강조: gradient에 bandpass 필터 적용
                grad_filtered = self._bandpass_filter(grad)

                delta.data += self.step_size * grad_filtered.sign()

                # ε-ball 투영
                delta.data = torch.clamp(delta.data, -self.epsilon * 2, self.epsilon * 2)

            delta.grad.zero_()

        # Step 3: 최종 적대적 이미지 생성
        with torch.no_grad():
            x_t_adv = x_t + delta
            adv_images = self._diffusion_reverse(x_t_adv, t)

            # 원본과의 차이를 ε 범위로 제한
            perturbation = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + perturbation, 0, 1)

        return adv_images.detach()

    def _bandpass_filter(self, x):
        """
        중주파 대역 강조 필터 (DiffAttack의 핵심 차별점)
        고주파는 억제하고 중주파에 에너지를 집중시킴
        """
        import torch.fft as fft

        f = fft.fft2(x)
        f_shift = fft.fftshift(f)

        h, w = x.shape[-2:]
        cy, cx = h // 2, w // 2

        y = torch.arange(h, device=x.device).float() - cy
        xx = torch.arange(w, device=x.device).float() - cx
        yy, xx = torch.meshgrid(y, xx, indexing="ij")
        dist = torch.sqrt(xx**2 + yy**2)

        max_r = min(cy, cx)
        r_low = max_r / 3
        r_high = max_r * 2 / 3

        # 중주파 대역만 통과 (bandpass)
        bandpass = ((dist > r_low) & (dist <= r_high)).float()

        # 약간의 저주파도 허용 (자연스러움 유지)
        bandpass += ((dist <= r_low) * 0.3).float()

        f_filtered = f_shift * bandpass
        x_filtered = torch.real(fft.ifft2(fft.ifftshift(f_filtered)))

        return x_filtered

    def attack_batch(self, dataloader, device):
        """
        전체 데이터에 대해 공격 수행

        Returns:
            dict: 원본/공격 이미지, 라벨, 예측 결과
        """
        self.model.eval()
        all_originals, all_advs, all_labels = [], [], []
        all_orig_preds, all_adv_preds = [], []

        for images, labels in dataloader:
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

        return {
            "originals": torch.cat(all_originals),
            "adversarials": torch.cat(all_advs),
            "labels": torch.cat(all_labels),
            "orig_preds": torch.cat(all_orig_preds),
            "adv_preds": torch.cat(all_adv_preds),
        }
