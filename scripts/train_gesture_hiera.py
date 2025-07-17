#!/usr/bin/env python3
"""
GESTURE Dataset Training with Hiera Architecture
Optimized for NVIDIA A10 GPU
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os

class GestureDataset(Dataset):
    def __init__(self, dataset_path, seizures_df, sequence_length=32, overlap=0.5, view='L'):
        self.dataset_path = Path(dataset_path)
        self.features_dir = self.dataset_path / 'features_fpc_8_fps_15'
        self.seizures_df = seizures_df
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.view = view
        
        # Filter valid seizures
        self.valid_seizures = self._filter_seizures()
        
        # Create windowed samples
        self.samples = self._create_windowed_samples()
        
        print(f"Dataset created: {len(self.samples)} samples from {len(self.valid_seizures)} seizures")
    
    def _filter_seizures(self):
        if self.view == 'L':
            mask = ~self.seizures_df['Discard'].isin(['Large', 'Yes'])
        else:
            mask = ~self.seizures_df['Discard'].isin(['Small', 'Yes'])
        return self.seizures_df[mask].copy()
    
    def _create_windowed_samples(self):
        samples = []
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for _, row in self.valid_seizures.iterrows():
            subject = row['Subject']
            seizure = row['Seizure']
            gtcs = row['GTCS']
            
            folder_name = f"{subject:03d}_{seizure:02d}_{self.view}"
            folder_path = self.features_dir / folder_name
            
            if not folder_path.exists():
                continue
            
            feature_files = sorted(folder_path.glob('*.pth'))
            
            if len(feature_files) < self.sequence_length:
                continue
            
            for start_idx in range(0, len(feature_files) - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                window_files = feature_files[start_idx:end_idx]
                
                samples.append({
                    'files': window_files,
                    'subject': subject,
                    'seizure': seizure,
                    'gtcs': gtcs,
                    'start_frame': start_idx,
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = []
        for file_path in sample['files']:
            try:
                feature = torch.load(file_path, map_location='cpu', weights_only=True)
                features.append(feature)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                features.append(torch.zeros(512))
        
        feature_sequence = torch.stack(features)
        label = torch.tensor(1 if sample['gtcs'] else 0, dtype=torch.long)
        
        return {
            'features': feature_sequence,
            'label': label,
            'subject': sample['subject'],
            'seizure': sample['seizure']
        }

class HieraSeizureClassifier(nn.Module):
    def __init__(self, input_dim=512, sequence_length=32, num_classes=2, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, hidden_dim) * 0.02)
        
        # Hierarchical transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.layers.append(layer)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

def main():
    print("=== GESTURE Dataset Training with Hiera Architecture ===")
    
    # Configuration - optimized for A10 GPU
    config = {
        'dataset_path': './gestures',
        'batch_size': 32,  # Increased for A10 GPU
        'sequence_length': 32,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Device: {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    seizures_df = pd.read_csv(os.path.join(config['dataset_path'], 'seizures.csv'))
    print(f"Loaded {len(seizures_df)} seizures")
    
    # Create train/val split by subjects
    subjects = seizures_df['Subject'].unique()
    train_subjects, val_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
    
    train_df = seizures_df[seizures_df['Subject'].isin(train_subjects)]
    val_df = seizures_df[seizures_df['Subject'].isin(val_subjects)]
    
    print(f"Train: {len(train_subjects)} subjects, {len(train_df)} seizures")
    print(f"Val: {len(val_subjects)} subjects, {len(val_df)} seizures")
    
    # Create datasets
    train_dataset = GestureDataset(config['dataset_path'], train_df, config['sequence_length'])
    val_dataset = GestureDataset(config['dataset_path'], val_df, config['sequence_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = HieraSeizureClassifier(
        input_dim=512,
        sequence_length=config['sequence_length'],
        num_classes=2,
        hidden_dim=256,
        num_layers=3
    ).to(config['device'])
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("-" * 60)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(config['device'])
            labels = batch['label'].to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config['device'])
                labels = batch['label'].to(config['device'])
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 60)
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
