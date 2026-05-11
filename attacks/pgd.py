"""
PGD (Projected Gradient Descent) 공격
- FGSM을 여러 번 반복하면서 ε-ball 안으로 투영
- 적대적 공격 연구의 표준 베이스라인 (Madry et al., 2018)
"""

import torch
import torch.nn as nn


class PGD:
    """PGD 적대적 공격"""

    def __init__(self, model, epsilon=0.04, step_size=0.01, num_steps=20,
                 criterion=None, random_start=True):
        """
        Args:
            model: 타겟 분류기
            epsilon: perturbation 크기 제한 (L-inf norm)
            step_size: 각 step의 이동 크기
            num_steps: 반복 횟수
            criterion: 손실 함수
            random_start: 랜덤 초기화 여부
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.random_start = random_start

    def attack(self, images, labels):
        """
        PGD 공격 수행

        Args:
            images: 원본 이미지 (B, C, H, W)
            labels: 정답 라벨 (B,)

        Returns:
            adv_images: 적대적 이미지 (B, C, H, W)
        """
        adv_images = images.clone().detach()

        # 랜덤 초기화: ε-ball 내 랜덤 시작점
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_images = torch.clamp(adv_images, -1, 1)

        for _ in range(self.num_steps):
            adv_images.requires_grad_(True)

            # Forward pass
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Step: gradient 부호 방향으로 step_size만큼 이동
            grad_sign = adv_images.grad.data.sign()
            adv_images = adv_images.detach() + self.step_size * grad_sign

            # ε-ball 투영: 원본 기준 ε 범위 내로 제한
            perturbation = torch.clamp(
                adv_images - images, min=-self.epsilon, max=self.epsilon
            )
            adv_images = torch.clamp(images + perturbation, -1, 1)

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
