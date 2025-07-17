import pandas as pd
import torch
from pathlib import Path

print("=== Dataset Test ===")
df = pd.read_csv("gestures/seizures.csv")
print(f"Dataset: {len(df)} seizures")
print(f"GTCS: {df['GTCS'].sum()}, Non-GTCS: {(~df['GTCS']).sum()}")

features_dir = Path("gestures/features_fpc_8_fps_15")
folders = list(features_dir.glob("*"))
print(f"Feature folders: {len(folders)}")

if folders:
    sample_files = list(folders[0].glob("*.pth"))
    if sample_files:
        feature = torch.load(sample_files[0], map_location="cpu", weights_only=True)
        print(f"Feature shape: {feature.shape}")
        print("✅ Dataset accessible")
