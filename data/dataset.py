import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CheXpertDataset(Dataset):
    """CheXpert 흉부 X-ray 데이터셋 (폐렴 Binary Classification)"""

    def __init__(self, csv_path, data_dir, transform=None):
        """
        Args:
            csv_path: CheXpert CSV 라벨 파일 경로
            data_dir: 이미지 루트 디렉토리
            transform: 이미지 변환
        """
        self.data_dir = data_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Frontal view만 사용 (Lateral 제외)
        df = df[df["Frontal/Lateral"] == "Frontal"]

        # Pneumonia 컬럼 기준 binary classification
        # 1.0 = Pneumonia(양성), 0.0 = Normal(음성)
        # -1.0 (uncertain) → 1.0으로 처리 (U-Ones 정책)
        # NaN → 제외 (단, No Finding=1이면 Negative)
        df = df[df["Pneumonia"].notna() | (df["No Finding"] == 1.0)]

        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(data_dir, row["Path"])
            pneumonia = row.get("Pneumonia", 0.0)

            # U-Ones: uncertain(-1) → positive(1)
            if pneumonia == -1.0:
                label = 1.0
            elif pneumonia == 1.0:
                label = 1.0
            else:
                label = 0.0

            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            image = self.transform(image)

        # grayscale → 3채널 복제 (pretrained 모델 호환)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def get_transforms(image_size=224, is_train=True):
    """학습/평가용 이미지 변환 파이프라인"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_dataloaders(csv_path, data_dir, image_size=224, batch_size=32,
                    num_workers=4, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Train/Val/Test DataLoader 생성"""

    full_dataset = CheXpertDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        transform=get_transforms(image_size, is_train=True),
    )

    total = len(full_dataset)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Val/Test는 augmentation 없는 transform 적용
    eval_transform = get_transforms(image_size, is_train=False)
    val_set.dataset = CheXpertDataset(csv_path, data_dir, transform=eval_transform)
    test_set.dataset = CheXpertDataset(csv_path, data_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    return train_loader, val_loader, test_loader
