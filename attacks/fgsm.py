"""
FGSM (Fast Gradient Sign Method) 공격
- 손실 함수의 gradient 부호 방향으로 ε만큼 한 번에 perturbation 추가
- 가장 단순하고 빠른 적대적 공격
"""

import torch
import torch.nn as nn


class FGSM:
    """FGSM 적대적 공격"""

    def __init__(self, model, epsilon=0.04, criterion=None):
        """
        Args:
            model: 타겟 분류기
            epsilon: perturbation 크기 제한
            criterion: 손실 함수 (기본: BCEWithLogitsLoss)
        """
        self.model = model
        self.epsilon = epsilon
        self.criterion = criterion or nn.BCEWithLogitsLoss()

    def attack(self, images, labels):
        """
        FGSM 공격 수행

        Args:
            images: 원본 이미지 (B, C, H, W)
            labels: 정답 라벨 (B,)

        Returns:
            adv_images: 적대적 이미지 (B, C, H, W)
        """
        images = images.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # FGSM: gradient 부호 방향으로 ε만큼 이동
        grad_sign = images.grad.data.sign()
        adv_images = images + self.epsilon * grad_sign

        # [0, 1] 범위로 클리핑
        adv_images = torch.clamp(adv_images, 0, 1)

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
