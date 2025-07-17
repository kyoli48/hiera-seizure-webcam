#!/usr/bin/env python3
"""
Improved GESTURE Dataset Training Script
Addresses overfitting issues with regularization and data augmentation
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import random
from scipy.stats import beta

class JitterAugmentation:
    """
    Beta distribution jittering from GESTURES paper
    """
    def __init__(self, gamma=4):
        self.gamma = gamma
        
    def __call__(self, sequence):
        """Apply jittering to sequence"""
        seq_len = len(sequence)
        if seq_len <= 1:
            return sequence
            
        # Generate jittering ratios using Beta distribution
        jitter_ratios = beta.rvs(self.gamma, self.gamma, size=seq_len)
        
        # Apply jittering
        jittered_sequence = []
        for i, feature in enumerate(sequence):
            # Add noise based on jitter ratio
            noise = torch.randn_like(feature) * 0.01 * jitter_ratios[i]
            jittered_feature = feature + noise
            jittered_sequence.append(jittered_feature)
            
        return jittered_sequence

class GestureDataset(Dataset):
    def __init__(self, dataset_path, seizures_df, sequence_length=16, overlap=0.5, 
                 view='L', augment=False, jitter_gamma=4):
        self.dataset_path = Path(dataset_path)
        self.features_dir = self.dataset_path / 'features_fpc_8_fps_15'
        self.seizures_df = seizures_df
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.view = view
        self.augment = augment
        
        # Initialize augmentation
        if augment:
            self.jitter_aug = JitterAugmentation(gamma=jitter_gamma)
        
        # Filter valid seizures
        self.valid_seizures = self._filter_seizures()
        
        # Create windowed samples
        self.samples = self._create_windowed_samples()
        
        print(f"Dataset created: {len(self.samples)} samples from {len(self.valid_seizures)} seizures")
        
        # Print class distribution
        gtcs_count = sum(1 for s in self.samples if s['gtcs'])
        print(f"Class distribution: GTCS={gtcs_count}, FOS={len(self.samples) - gtcs_count}")
    
    def _filter_seizures(self):
        if self.view == 'L':
            mask = ~self.seizures_df['Discard'].isin(['Large', 'Yes'])
        else:
            mask = ~self.seizures_df['Discard'].isin(['Small', 'Yes'])
        return self.seizures_df[mask].copy()
    
    def _create_windowed_samples(self):
        samples = []
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
        
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
            
            # Create multiple samples per seizure
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
        
        # Apply augmentation if enabled
        if self.augment:
            features = self.jitter_aug(features)
        
        # Add random temporal dropout (randomly skip some frames)
        if self.augment and random.random() < 0.3:
            skip_indices = random.sample(range(len(features)), k=max(1, len(features) // 8))
            for idx in skip_indices:
                features[idx] = torch.zeros_like(features[idx])
        
        feature_sequence = torch.stack(features)
        label = torch.tensor(1 if sample['gtcs'] else 0, dtype=torch.long)
        
        return {
            'features': feature_sequence,
            'label': label,
            'subject': sample['subject'],
            'seizure': sample['seizure']
        }

class ImprovedSeizureClassifier(nn.Module):
    """
    Improved model with better regularization and reduced complexity
    """
    def __init__(self, input_dim=512, sequence_length=16, num_classes=2, 
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM (as in GESTURES paper)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2,  # Smaller hidden size since bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for better feature aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classification head with strong regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention-based aggregation
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        x = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def main():
    print("=== Improved GESTURE Dataset Training ===")
    
    # Configuration - reduced complexity to prevent overfitting
    config = {
        'dataset_path': './gestures',
        'batch_size': 16,  # Reduced batch size for better generalization
        'sequence_length': 16,  # Reduced from 32 to 16 (as in GESTURES paper)
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,  # Increased weight decay
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'hidden_dim': 128,  # Reduced from 256
        'num_layers': 2,    # Reduced from 3
        'dropout': 0.3,     # Increased dropout
        'jitter_gamma': 4,  # From GESTURES paper
        'patience': 7,      # Early stopping patience
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
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=0.2, random_state=42, stratify=None
    )
    
    train_df = seizures_df[seizures_df['Subject'].isin(train_subjects)]
    val_df = seizures_df[seizures_df['Subject'].isin(val_subjects)]
    
    print(f"Train: {len(train_subjects)} subjects, {len(train_df)} seizures")
    print(f"Val: {len(val_subjects)} subjects, {len(val_df)} seizures")
    
    # Create datasets with augmentation for training
    train_dataset = GestureDataset(
        config['dataset_path'], 
        train_df, 
        config['sequence_length'],
        augment=True,
        jitter_gamma=config['jitter_gamma']
    )
    val_dataset = GestureDataset(
        config['dataset_path'], 
        val_df, 
        config['sequence_length'],
        augment=False  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Compute class weights for balanced training
    all_labels = [sample['gtcs'] for sample in train_dataset.samples]
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(all_labels), 
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config['device'])
    print(f"Class weights: {class_weights}")
    
    # Create model
    model = ImprovedSeizureClassifier(
        input_dim=512,
        sequence_length=config['sequence_length'],
        num_classes=2,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Use ReduceLROnPlateau instead of CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True,
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("-" * 60)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
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
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 60)
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved as 'best_model.pth'")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()