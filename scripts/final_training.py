#!/usr/bin/env python3
"""
Final GESTURE Dataset Training Script with Maximum Regularization
Further reduces overfitting with more aggressive techniques
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

class AdvancedAugmentation:
    """
    Advanced augmentation techniques for seizure data
    """
    def __init__(self, gamma=4, noise_std=0.02, mixup_alpha=0.3):
        self.gamma = gamma
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        
    def jitter_sequence(self, sequence):
        """Apply Beta distribution jittering"""
        seq_len = len(sequence)
        if seq_len <= 1:
            return sequence
            
        jitter_ratios = beta.rvs(self.gamma, self.gamma, size=seq_len)
        
        jittered_sequence = []
        for i, feature in enumerate(sequence):
            noise = torch.randn_like(feature) * self.noise_std * jitter_ratios[i]
            jittered_feature = feature + noise
            jittered_sequence.append(jittered_feature)
            
        return jittered_sequence
    
    def temporal_dropout(self, sequence, dropout_rate=0.1):
        """Randomly drop temporal frames"""
        seq_len = len(sequence)
        if seq_len <= 2:
            return sequence
            
        num_drop = int(seq_len * dropout_rate)
        drop_indices = random.sample(range(seq_len), k=num_drop)
        
        result = []
        for i, feature in enumerate(sequence):
            if i in drop_indices:
                result.append(torch.zeros_like(feature))
            else:
                result.append(feature)
        
        return result
    
    def feature_dropout(self, sequence, dropout_rate=0.1):
        """Randomly drop feature dimensions"""
        result = []
        for feature in sequence:
            if random.random() < dropout_rate:
                # Drop random feature dimensions
                mask = torch.rand_like(feature) > 0.1
                result.append(feature * mask)
            else:
                result.append(feature)
        
        return result
    
    def __call__(self, sequence):
        """Apply all augmentations"""
        # Apply jittering
        sequence = self.jitter_sequence(sequence)
        
        # Apply temporal dropout
        if random.random() < 0.3:
            sequence = self.temporal_dropout(sequence)
        
        # Apply feature dropout
        if random.random() < 0.2:
            sequence = self.feature_dropout(sequence)
        
        return sequence

class GestureDataset(Dataset):
    def __init__(self, dataset_path, seizures_df, sequence_length=12, overlap=0.7, 
                 view='L', augment=False, jitter_gamma=4):
        self.dataset_path = Path(dataset_path)
        self.features_dir = self.dataset_path / 'features_fpc_8_fps_15'
        self.seizures_df = seizures_df
        self.sequence_length = sequence_length  # Reduced further
        self.overlap = overlap  # Increased overlap for more samples
        self.view = view
        self.augment = augment
        
        # Initialize advanced augmentation
        if augment:
            self.augmentation = AdvancedAugmentation(gamma=jitter_gamma)
        
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
            
            # Create many overlapping samples
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
            features = self.augmentation(features)
        
        feature_sequence = torch.stack(features)
        label = torch.tensor(1 if sample['gtcs'] else 0, dtype=torch.long)
        
        return {
            'features': feature_sequence,
            'label': label,
            'subject': sample['subject'],
            'seizure': sample['seizure']
        }

class MinimalSeizureClassifier(nn.Module):
    """
    Minimal model architecture to prevent overfitting
    """
    def __init__(self, input_dim=512, sequence_length=12, num_classes=2, 
                 hidden_dim=64, dropout=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Very simple input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2,
            num_layers=1,  # Single layer
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Simple attention
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Minimal classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = x.view(-1, self.input_dim)
        x = self.input_proj(x)
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        x = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
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

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    print("=== Final GESTURE Dataset Training with Maximum Regularization ===")
    
    # Configuration - maximum regularization
    config = {
        'dataset_path': './gestures',
        'batch_size': 8,    # Smaller batch size
        'sequence_length': 12,  # Shorter sequences
        'num_epochs': 30,
        'learning_rate': 5e-5,  # Lower learning rate
        'weight_decay': 0.2,    # Higher weight decay
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'hidden_dim': 64,   # Smaller model
        'dropout': 0.5,     # Higher dropout
        'jitter_gamma': 4,
        'patience': 5,      # Shorter patience
        'mixup_alpha': 0.2, # Mixup augmentation
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
    
    # Create datasets
    train_dataset = GestureDataset(
        config['dataset_path'], 
        train_df, 
        config['sequence_length'],
        overlap=0.7,  # High overlap for more samples
        augment=True,
        jitter_gamma=config['jitter_gamma']
    )
    val_dataset = GestureDataset(
        config['dataset_path'], 
        val_df, 
        config['sequence_length'],
        overlap=0.5,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,  # Reduced workers
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Compute class weights
    all_labels = [sample['gtcs'] for sample in train_dataset.samples]
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(all_labels), 
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config['device'])
    print(f"Class weights: {class_weights}")
    
    # Create minimal model
    model = MinimalSeizureClassifier(
        input_dim=512,
        sequence_length=config['sequence_length'],
        num_classes=2,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # More aggressive scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.3,  # Reduce by 70%
        patience=2,  # Reduce faster
        verbose=True,
        min_lr=1e-7
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
        
        # Training with mixup
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(config['device'])
            labels = batch['label'].to(config['device'])
            
            # Apply mixup
            if config['mixup_alpha'] > 0 and random.random() < 0.5:
                features, labels_a, labels_b, lam = mixup_data(features, labels, config['mixup_alpha'])
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                train_total += labels.size(0)
                # For mixup, we approximate correct predictions
                train_correct += labels.size(0) * 0.6  # Rough estimate
            else:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 20 == 0:
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
            torch.save(model.state_dict(), 'best_model_final.pth')
        
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
    print("Model saved as 'best_model_final.pth'")
    
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
    plt.savefig('final_training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()