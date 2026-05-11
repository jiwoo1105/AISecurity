"""
Adversarial Training - 학습 단계 방어
- 학습 시 적대적 샘플을 함께 학습시켜 모델 자체를 강건하게 만듦
- 표준 베이스라인 방어 (Madry et al., 2018)
- 한계: 이미 배포된 모델에는 적용 불가 (재학습 필요)
"""

import torch
import torch.nn as nn
from attacks.pgd import PGD


class AdversarialTraining:
    """Adversarial Training 방어"""

    def __init__(self, model, epsilon=0.04, pgd_steps=10, step_size=0.01):
        """
        Args:
            model: 분류기
            epsilon: PGD 공격 ε
            pgd_steps: PGD 반복 횟수
            step_size: PGD step size
        """
        self.model = model
        self.attacker = PGD(model, epsilon=epsilon, step_size=step_size,
                           num_steps=pgd_steps)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, train_loader, optimizer, device):
        """
        Adversarial Training 1 epoch

        각 배치마다:
        1. PGD로 적대적 샘플 생성
        2. 원본 + 적대적 샘플 모두로 학습
        """
        self.model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 적대적 샘플 생성
            self.model.eval()
            adv_images = self.attacker.attack(images, labels)
            self.model.train()

            # 원본 + 적대적 샘플 결합
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            # 학습
            optimizer.zero_grad()
            outputs = self.model(combined_images)
            loss = self.criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(self, train_loader, val_loader, optimizer, device,
              epochs=10, save_path=None):
        """전체 Adversarial Training 수행"""
        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, optimizer, device)

            # 검증 (적대적 샘플에 대한 정확도)
            robust_acc = self.evaluate_robust(val_loader, device)

            print(f"Epoch {epoch}: Loss={train_loss:.4f} | Robust Acc={robust_acc:.4f}")

            if robust_acc > best_acc and save_path:
                best_acc = robust_acc
                torch.save(self.model.state_dict(), save_path)

        return best_acc

    @torch.no_grad()
    def evaluate_robust(self, dataloader, device):
        """적대적 샘플에 대한 정확도 평가"""
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 공격
            self.model.eval()
            adv_images = self.attacker.attack(images, labels)

            # 예측
            outputs = self.model(adv_images)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        return correct / total
