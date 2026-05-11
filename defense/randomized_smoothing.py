"""
Randomized Smoothing - 추론 단계 방어
- 입력에 가우시안 노이즈를 여러 번 추가하고 다수결로 판단
- 수학적으로 검증된 인증 방어 (certified defense)
- Cohen et al., 2019

참고: 분류기 자체를 수정하지 않음 (플러그인 방식)
"""

import torch
import torch.nn.functional as F


class RandomizedSmoothing:
    """Randomized Smoothing 방어"""

    def __init__(self, model, sigma=0.25, n_samples=100):
        """
        Args:
            model: 분류기
            sigma: 가우시안 노이즈 표준편차
            n_samples: 노이즈 샘플링 횟수 (많을수록 정확, 느림)
        """
        self.model = model
        self.sigma = sigma
        self.n_samples = n_samples

    def predict(self, images, device):
        """
        Randomized Smoothing 예측

        1. 같은 이미지에 n_samples번 서로 다른 노이즈 추가
        2. 각각 분류기로 예측
        3. 다수결로 최종 판단

        Args:
            images: 입력 이미지 (B, C, H, W)

        Returns:
            preds: 최종 예측 (B,)
            probs: 평균 확률 (B,)
        """
        self.model.eval()
        batch_size = images.shape[0]

        # n_samples번 노이즈 추가 + 예측
        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                noise = torch.randn_like(images) * self.sigma
                noisy_images = torch.clamp(images + noise, 0, 1)

                logits = self.model(noisy_images)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)

        # 평균 확률로 다수결
        avg_probs = torch.stack(all_probs).mean(dim=0)
        preds = (avg_probs >= 0.5).long()

        return preds, avg_probs

    def predict_batch(self, dataloader, device):
        """
        전체 데이터에 대해 Randomized Smoothing 예측

        Returns:
            dict: 예측 결과, 라벨
        """
        all_preds, all_probs, all_labels = [], [], []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            preds, probs = self.predict(images, device)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        return {
            "preds": torch.cat(all_preds),
            "probs": torch.cat(all_probs),
            "labels": torch.cat(all_labels),
        }
