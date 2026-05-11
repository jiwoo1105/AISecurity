"""
정상 X-ray FFT 통계 사전 계산
- CheXpert에서 No Finding=1 (정상) 이미지만 추출
- 주파수 대역별 평균/표준편차 계산
- data/fft_statistics.pkl로 저장
"""

import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import CheXpertDataset, get_transforms
from torch.utils.data import DataLoader
from defense.fft_analyzer import compute_normal_stats
import pandas as pd


class NormalOnlyDataset(torch.utils.data.Dataset):
    """No Finding=1인 정상 X-ray만 추출"""

    def __init__(self, csv_path, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        # Frontal + No Finding=1 (확실한 정상)만
        df = df[df["Frontal/Lateral"] == "Frontal"]
        df = df[df["No Finding"] == 1.0]

        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(data_dir, row["Path"])
            self.samples.append(img_path)

        print(f"정상 X-ray: {len(self.samples)}장")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.samples[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, 0  # 라벨은 0 (정상)


def main():
    csv_path = "data/chexpert/train.csv"
    data_dir = "data/chexpert"
    save_path = "data/fft_statistics.pkl"

    device = "cpu"  # FFT는 CPU로 충분

    transform = get_transforms(224, is_train=False)
    dataset = NormalOnlyDataset(csv_path, data_dir, transform)

    # 전체 다 쓰면 느리니까 최대 2000장만 샘플링
    max_samples = min(2000, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(max_samples))

    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

    print(f"FFT 통계 계산 중 ({max_samples}장)...")
    stats = compute_normal_stats(loader, device)

    print(f"\n=== 정상 X-ray FFT 통계 ===")
    for band, values in stats.items():
        print(f"  {band}: mean={values['mean']:.4f}, std={values['std']:.4f}")

    torch.save(stats, save_path)
    print(f"\n저장 완료: {save_path}")


if __name__ == "__main__":
    main()
