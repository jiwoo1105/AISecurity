"""
베이스라인 분류기: ResNet-50, DenseNet-121
ImageNet pretrained → CheXpert 폐렴 분류 fine-tuning
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_classifier(name="resnet50", num_classes=2, pretrained=True):
    """
    분류기 생성

    Args:
        name: 모델 이름 ("resnet50" | "densenet121")
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 가중치 사용 여부

    Returns:
        PyTorch 모델
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        # 첫 번째 conv layer: 3ch 입력 유지 (grayscale을 3ch로 복제하므로)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )

    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier.in_features, num_classes),
        )

    else:
        raise ValueError(f"Unsupported model: {name}. Use 'resnet50' or 'densenet121'.")

    return model


class PneumoniaClassifier(nn.Module):
    """폐렴 분류기 래퍼 (binary classification)"""

    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.model = get_classifier(name, num_classes=1, pretrained=pretrained)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logit = self.model(x).squeeze(-1)
        return logit

    def predict_proba(self, x):
        """확률 출력 (ECE 계산용)"""
        logit = self.forward(x)
        return self.sigmoid(logit)

    def predict(self, x, threshold=0.5):
        """이진 예측"""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()
