"""
CheXpert 데이터셋 전처리 유틸리티
- CSV 라벨 파싱 및 정리
- 정상/폐렴 라벨 통계 확인
- 클래스 불균형 대응 (가중치 계산)
"""

import pandas as pd
import numpy as np
from collections import Counter


def parse_chexpert_labels(csv_path, target="Pneumonia"):
    """
    CheXpert CSV에서 폐렴 binary classification용 라벨 추출

    라벨 정책 (U-Ones):
        1.0  → Positive (폐렴)
       -1.0  → Positive (uncertain → positive)
        0.0  → Negative (정상)
        NaN  → 제외 (단, No Finding=1이면 Negative)
    """
    df = pd.read_csv(csv_path)

    labels = []
    valid_indices = []

    for idx, row in df.iterrows():
        pneumonia = row.get(target, float("nan"))
        no_finding = row.get("No Finding", 0.0)

        if pd.isna(pneumonia):
            if no_finding == 1.0:
                labels.append(0)
                valid_indices.append(idx)
            continue

        if pneumonia == -1.0 or pneumonia == 1.0:
            labels.append(1)
        else:
            labels.append(0)
        valid_indices.append(idx)

    filtered_df = df.iloc[valid_indices].copy()
    filtered_df["binary_label"] = labels

    return filtered_df


def compute_class_weights(labels):
    """클래스 불균형 대응용 가중치 계산"""
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return weights


def print_dataset_stats(csv_path):
    """데이터셋 통계 출력"""
    df = parse_chexpert_labels(csv_path)
    counts = df["binary_label"].value_counts()

    print("=" * 40)
    print("CheXpert Dataset Statistics")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"  Normal  (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Pneumonia(1): {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")

    weights = compute_class_weights(df["binary_label"].tolist())
    print(f"\nClass weights: {weights}")
    print("=" * 40)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_dataset_stats(sys.argv[1])
    else:
        print("Usage: python preprocessing.py <chexpert_csv_path>")
