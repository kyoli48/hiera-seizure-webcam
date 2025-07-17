import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models.video import r2plus1d_18
import warnings
warnings.filterwarnings('ignore')

def load_timestamps(timestamps_file="raw_videos/time_stamps.xlsx"):
    """Load seizure timestamps from Excel file"""
    try:
        df = pd.read_excel(timestamps_file)
        print(f"Loaded timestamps for {len(df)} videos")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Warning: Could not load timestamps file: {e}")
        return None

def parse_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    if pd.isna(time_str) or time_str == "":
        return None
    
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        
        if len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return float(time_str)  # Assume it's already in seconds
    except:
        print(f"Warning: Could not parse time '{time_str}'")
        return None

# Your actual model architecture from final_training.py
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

class VideoProcessor:
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load feature extractor (R2+1D pretrained on Kinetics)
        self.feature_extractor = r2plus1d_18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Video preprocessing (based on your project's requirements)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Standard size for pretrained models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Video processor initialized on {self.device}")
    
    def load_and_preprocess_video(self, video_path, target_fps=30):
        """
        Load video and preprocess according to your project requirements:
        1. Unified frame rate to 30 FPS
        2. Resize to 224x224 pixels
        3. Normalize pixel values
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Original FPS: {original_fps}, Total frames: {total_frames}")
        
        frames = []
        frame_count = 0
        
        # Calculate frame sampling to achieve target FPS
        if original_fps > target_fps:
            frame_step = int(original_fps / target_fps)
        else:
            frame_step = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to achieve target FPS
            if frame_count % frame_step == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        print(f"Extracted {len(frames)} frames at ~{target_fps} FPS")
        return np.array(frames)
    
    def extract_seizure_segment(self, video_path, start_time=None, end_time=None, clip_length=8, stride=4):
        """
        Extract features from a specific seizure segment instead of the entire video
        
        Args:
            video_path: Path to video file
            start_time: Seizure start time in seconds (None = start from beginning)
            end_time: Seizure end time in seconds (None = process until end)
            clip_length: Frames per clip for feature extraction
            stride: Stride between clips
        """
        # Load and preprocess video
        frames = self.load_and_preprocess_video(video_path)
        
        # Convert time to frame indices (assuming 30 FPS after preprocessing)
        fps = 30
        start_frame = int(start_time * fps) if start_time is not None else 0
        end_frame = int(end_time * fps) if end_time is not None else len(frames)
        
        # Clip to valid range
        start_frame = max(0, start_frame)
        end_frame = min(len(frames), end_frame)
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid time range: start={start_time}s, end={end_time}s")
        
        # Extract seizure segment
        seizure_frames = frames[start_frame:end_frame]
        print(f"Extracted seizure segment: frames {start_frame}-{end_frame} ({len(seizure_frames)} frames, {len(seizure_frames)/fps:.1f}s)")
        
        # Preprocess frames
        preprocessed_frames = []
        for frame in seizure_frames:
            frame_tensor = self.transform(frame)
            preprocessed_frames.append(frame_tensor)
        
        preprocessed_frames = torch.stack(preprocessed_frames)
        
        # Extract features using sliding window
        all_features = []
        num_frames = len(preprocessed_frames)
        
        if num_frames < clip_length:
            print(f"Warning: Seizure segment too short ({num_frames} frames), padding with zeros")
            # Pad segment if too short
            padding = torch.zeros(clip_length - num_frames, 3, 224, 224)
            preprocessed_frames = torch.cat([preprocessed_frames, padding], dim=0)
            num_frames = clip_length
        
        for start_idx in range(0, num_frames - clip_length + 1, stride):
            end_idx = start_idx + clip_length
            clip = preprocessed_frames[start_idx:end_idx]
            
            # Rearrange for video model: (batch, channels, time, height, width)
            clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(clip)
            
            all_features.append(features.cpu().squeeze(0))
        
        if len(all_features) == 0:
            raise ValueError(f"No features extracted from seizure segment")
        
        video_features = torch.stack(all_features)
        print(f"Extracted features shape: {video_features.shape}")
        
        return video_features
    
    def extract_features_from_video(self, video_path, clip_length=8, stride=4):
        """Extract features from full video using sliding window (fallback method)"""
        
        # Load and preprocess video
        frames = self.load_and_preprocess_video(video_path)
        
        # Preprocess frames
        preprocessed_frames = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            preprocessed_frames.append(frame_tensor)
        
        preprocessed_frames = torch.stack(preprocessed_frames)
        
        # Extract features using sliding window
        all_features = []
        num_frames = len(preprocessed_frames)
        
        if num_frames < clip_length:
            print(f"Warning: Video too short ({num_frames} frames), padding with zeros")
            # Pad video if too short
            padding = torch.zeros(clip_length - num_frames, 3, 224, 224)
            preprocessed_frames = torch.cat([preprocessed_frames, padding], dim=0)
            num_frames = clip_length
        
        for start_idx in range(0, num_frames - clip_length + 1, stride):
            end_idx = start_idx + clip_length
            clip = preprocessed_frames[start_idx:end_idx]
            
            # Rearrange for video model: (batch, channels, time, height, width)
            clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(clip)
            
            all_features.append(features.cpu().squeeze(0))
        
        if len(all_features) == 0:
            raise ValueError(f"No features extracted from video: {video_path}")
        
        video_features = torch.stack(all_features)
        print(f"Extracted features shape: {video_features.shape}")
        
        return video_features

class SeizureInferenceEngine:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load trained classifier
        self.classifier = self._load_classifier(model_path)
        
        # Initialize video processor
        self.video_processor = VideoProcessor(device=self.device)
        
        print(f"Inference engine ready on {self.device}")
    
    def _load_classifier(self, model_path):
        # Use the exact same parameters as your final training configuration
        model = MinimalSeizureClassifier(
            input_dim=512,  # R2+1D features
            sequence_length=12,  # Your training used sequence_length=12
            num_classes=2,
            hidden_dim=64,  # Your training used hidden_dim=64
            dropout=0.5  # Your training used dropout=0.5
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def predict_single_video(self, video_path, timestamps_df=None):
        """Run complete inference pipeline on a single video with optional timestamps"""
        
        try:
            video_name = Path(video_path).stem
            
            # Check if we have timestamp information for this video
            seizure_times = self._get_seizure_timestamps(video_name, timestamps_df)
            
            if seizure_times:
                print(f"Processing: {video_path}")
                print(f"Found seizure timestamps: {seizure_times}")
                
                # Process each seizure segment separately
                segment_results = []
                
                for i, (start_time, end_time) in enumerate(seizure_times):
                    print(f"\nProcessing seizure segment {i+1}: {start_time}s - {end_time}s")
                    
                    try:
                        # Extract features from seizure segment only
                        features = self.video_processor.extract_seizure_segment(
                            video_path, start_time, end_time
                        )
                        
                        # Apply temporal segmentation
                        segment_result = self._temporal_segment_inference(features, video_path, segment_id=i+1)
                        segment_result.update({
                            'seizure_start_time': start_time,
                            'seizure_end_time': end_time,
                            'seizure_duration': end_time - start_time,
                            'segment_id': i+1
                        })
                        segment_results.append(segment_result)
                        
                    except Exception as e:
                        print(f"Error processing segment {i+1}: {str(e)}")
                        segment_results.append({
                            'segment_id': i+1,
                            'seizure_start_time': start_time,
                            'seizure_end_time': end_time,
                            'error': str(e),
                            'prediction': -1
                        })
                
                # Aggregate results from all segments
                results = self._aggregate_segment_results(segment_results, video_path)
                
            else:
                # No timestamps available, process entire video
                print(f"Processing: {video_path} (no timestamps available - processing full video)")
                features = self.video_processor.extract_features_from_video(video_path)
                results = self._temporal_segment_inference(features, video_path)
                results['analysis_type'] = 'full_video'
            
            return results
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return {
                'video_path': video_path,
                'error': str(e),
                'prediction': -1,
                'confidence': 0.0,
                'seizure_type': 'ERROR'
            }
    
    def _get_seizure_timestamps(self, video_name, timestamps_df):
        """Extract seizure timestamps for a specific video from your Excel format"""
        if timestamps_df is None:
            return None
        
        try:
            # Read Excel file properly handling your format
            import pandas as pd
            
            # Try to load Excel file again with proper handling
            excel_data = pd.read_excel("raw_videos/time_stamps.xlsx")
            
            # Your format has columns: video_name, start_time, end_time, notes, plus readable time columns
            video_names_to_try = [video_name, video_name + '.mp4', video_name + '.avi']
            
            seizure_times = []
            current_video = None
            
            for idx, row in excel_data.iterrows():
                # Check if this row has a video name
                if pd.notna(row.iloc[0]) and row.iloc[0] != "":
                    current_video = str(row.iloc[0]).strip()
                
                # Check if current video matches what we're looking for
                if current_video in video_names_to_try:
                    # Extract time information - your file has readable times in columns 5 and 6
                    if len(row) > 6:  # Make sure we have enough columns
                        start_str = str(row.iloc[5]) if pd.notna(row.iloc[5]) else None
                        end_str = str(row.iloc[6]) if pd.notna(row.iloc[6]) else None
                        
                        if start_str and end_str and start_str != 'nan' and end_str != 'nan':
                            start_time = self._parse_time_string(start_str)
                            end_time = self._parse_time_string(end_str)
                            
                            if start_time is not None and end_time is not None:
                                seizure_times.append((start_time, end_time))
                                print(f"Found seizure: {start_str} - {end_str} ({start_time:.1f}s - {end_time:.1f}s)")
            
            return seizure_times if seizure_times else None
            
        except Exception as e:
            print(f"Error parsing timestamps for {video_name}: {e}")
            return None
    
    def _parse_time_string(self, time_str):
        """Parse time string in MM:SS format to seconds"""
        try:
            time_str = str(time_str).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return None
    
    def _aggregate_segment_results(self, segment_results, video_path):
        """Aggregate results from multiple seizure segments"""
        
        valid_segments = [r for r in segment_results if 'error' not in r]
        
        if not valid_segments:
            return {
                'video_path': video_path,
                'error': 'All segments failed',
                'prediction': -1,
                'confidence': 0.0,
                'seizure_type': 'ERROR',
                'analysis_type': 'seizure_segments'
            }
        
        # Aggregate predictions using majority vote and average confidence
        predictions = [r['prediction'] for r in valid_segments]
        confidences = [r['confidence'] for r in valid_segments]
        
        # Majority vote for final prediction
        final_prediction = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean(confidences)
        
        # Calculate per-segment statistics
        tcs_segments = sum(1 for p in predictions if p == 1)
        total_segments = len(valid_segments)
        
        return {
            'video_path': video_path,
            'prediction': final_prediction,
            'seizure_type': 'TCS' if final_prediction == 1 else 'FOS',
            'confidence': avg_confidence,
            'analysis_type': 'seizure_segments',
            'total_seizure_segments': total_segments,
            'tcs_segments': tcs_segments,
            'fos_segments': total_segments - tcs_segments,
            'segment_details': segment_results,
            'seizure_summary': {
                'total_seizure_time': sum(r.get('seizure_duration', 0) for r in valid_segments),
                'avg_segment_confidence': avg_confidence,
                'segment_predictions': predictions
            }
        }
    
    def _temporal_segment_inference(self, video_features, video_path, segment_id=None):
        """Apply temporal segment network inference"""
        
        # Adaptive segmentation based on video length
        num_frames = video_features.shape[0]
        min_segments = 8   # Reduced for seizure segments
        max_segments = 20  # Reduced for seizure segments
        sequence_length = 12  # Match your training sequence length
        
        num_segments = min(max(min_segments, num_frames // 4), max_segments)
        
        # Sample segments across the video
        if num_frames >= sequence_length:
            segment_indices = np.linspace(0, num_frames - sequence_length, num_segments, dtype=int)
            sampled_segments = []
            
            for start_idx in segment_indices:
                end_idx = start_idx + sequence_length
                if end_idx <= num_frames:
                    segment = video_features[start_idx:end_idx]
                else:
                    # Pad if necessary
                    segment = video_features[start_idx:]
                    padding_needed = sequence_length - segment.shape[0]
                    padding = torch.zeros(padding_needed, video_features.shape[1])
                    segment = torch.cat([segment, padding], dim=0)
                
                sampled_segments.append(segment)
            
            # Stack segments - this creates shape (num_segments, sequence_length, feature_dim)
            segment_features = torch.stack(sampled_segments)  # Remove the extra unsqueeze
            
            # The model expects input shape (batch_size, sequence_length, feature_dim)
            # We need to process each segment separately and then aggregate
            all_predictions = []
            all_probabilities = []
            
            for i in range(segment_features.shape[0]):
                single_segment = segment_features[i:i+1].to(self.device)  # Shape: (1, seq_len, feat_dim)
                
                with torch.no_grad():
                    outputs = self.classifier(single_segment)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                all_predictions.append(predicted.cpu().item())
                all_probabilities.append(probabilities.cpu().numpy()[0])
            
            # Aggregate predictions using majority vote
            final_prediction = max(set(all_predictions), key=all_predictions.count)
            avg_probabilities = np.mean(all_probabilities, axis=0)
            confidence = float(np.max(avg_probabilities))
            
        else:
            # Video too short, pad to minimum length
            padding_needed = sequence_length - num_frames
            padding = torch.zeros(padding_needed, video_features.shape[1])
            padded_features = torch.cat([video_features, padding], dim=0)
            segment_features = padded_features.unsqueeze(0).to(self.device)  # Shape: (1, seq_len, feat_dim)
            
            with torch.no_grad():
                outputs = self.classifier(segment_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            final_prediction = predicted.cpu().item()
            avg_probabilities = probabilities.cpu().numpy()[0]
            confidence = float(torch.max(probabilities).cpu())
        
        # Prepare results
        segment_suffix = f" (segment {segment_id})" if segment_id else ""
        
        return {
            'video_path': video_path + segment_suffix,
            'prediction': final_prediction,
            'seizure_type': 'TCS' if final_prediction == 1 else 'FOS',
            'confidence': confidence,
            'probabilities': {
                'FOS': float(avg_probabilities[0]),
                'TCS': float(avg_probabilities[1])
            },
            'num_segments_used': num_segments,
            'video_length_clips': num_frames
        }
    
    def process_video_directory(self, video_dir, output_dir, timestamps_file=None):
        """Process all videos in a directory with optional timestamp information"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load timestamps if provided
        timestamps_df = None
        if timestamps_file and os.path.exists(timestamps_file):
            timestamps_df = load_timestamps(timestamps_file)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
            video_files.extend(Path(video_dir).glob(f'*{ext.upper()}'))
        
        video_files = [str(f) for f in video_files]
        
        if len(video_files) == 0:
            print(f"No video files found in {video_dir}")
            return []
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process each video
        all_results = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            result = self.predict_single_video(video_path, timestamps_df)
            all_results.append(result)
            
            # Print result
            if 'error' not in result:
                analysis_type = result.get('analysis_type', 'unknown')
                if analysis_type == 'seizure_segments':
                    total_segments = result.get('total_seizure_segments', 0)
                    tcs_segments = result.get('tcs_segments', 0)
                    print(f"✓ {Path(video_path).name}: {result['seizure_type']} (confidence: {result['confidence']:.3f}) - {total_segments} seizure segments, {tcs_segments} TCS")
                else:
                    print(f"✓ {Path(video_path).name}: {result['seizure_type']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"✗ {Path(video_path).name}: ERROR - {result['error']}")
        
        # Save results
        self._save_results(all_results, output_dir)
        
        return all_results
    
    def _save_results(self, results, output_dir):
        """Save inference results with timestamp-aware formatting"""
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, 'detailed_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for result in results:
            if 'error' not in result:
                base_data = {
                    'video_name': Path(result['video_path']).name,
                    'prediction': result['prediction'],
                    'seizure_type': result['seizure_type'],
                    'confidence': result['confidence'],
                    'analysis_type': result.get('analysis_type', 'unknown')
                }
                
                if result.get('analysis_type') == 'seizure_segments':
                    # Enhanced data for timestamp-based analysis
                    base_data.update({
                        'total_seizure_segments': result.get('total_seizure_segments', 0),
                        'tcs_segments': result.get('tcs_segments', 0),
                        'fos_segments': result.get('fos_segments', 0),
                        'total_seizure_time': result.get('seizure_summary', {}).get('total_seizure_time', 0),
                        'avg_segment_confidence': result.get('seizure_summary', {}).get('avg_segment_confidence', 0)
                    })
                else:
                    # Standard full-video analysis
                    base_data.update({
                        'fos_probability': result.get('probabilities', {}).get('FOS', 0),
                        'tcs_probability': result.get('probabilities', {}).get('TCS', 0),
                        'segments_used': result.get('num_segments_used', 0)
                    })
                
                summary_data.append(base_data)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_dir, 'summary_results.csv')
            df.to_csv(csv_path, index=False)
            
            # Print enhanced summary statistics
            timestamp_videos = len(df[df['analysis_type'] == 'seizure_segments'])
            full_videos = len(df[df['analysis_type'] == 'full_video'])
            
            tcs_count = len(df[df['seizure_type'] == 'TCS'])
            fos_count = len(df[df['seizure_type'] == 'FOS'])
            avg_confidence = df['confidence'].mean()
            
            print(f"\n=== ENHANCED SUMMARY ===")
            print(f"Total videos processed: {len(df)}")
            print(f"Videos with timestamps: {timestamp_videos}")
            print(f"Videos processed fully: {full_videos}")
            print(f"TCS predictions: {tcs_count}")
            print(f"FOS predictions: {fos_count}")
            print(f"Average confidence: {avg_confidence:.3f}")
            
            if timestamp_videos > 0:
                total_seizure_segments = df[df['analysis_type'] == 'seizure_segments']['total_seizure_segments'].sum()
                total_tcs_segments = df[df['analysis_type'] == 'seizure_segments']['tcs_segments'].sum()
                print(f"Total seizure segments analyzed: {total_seizure_segments}")
                print(f"TCS seizure segments: {total_tcs_segments}")
                print(f"FOS seizure segments: {total_seizure_segments - total_tcs_segments}")
            
            print(f"Results saved to: {output_dir}")

# Main execution function
def main():
    # Configuration
    VIDEO_DIR = "raw_videos"                    # Directory containing your videos
    MODEL_PATH = "best_model_final.pth"               # Path to your trained model
    OUTPUT_DIR = "inference_results"            # Where to save results
    TIMESTAMPS_FILE = "raw_videos/time_stamps.xlsx"  # Path to your timestamps file
    
    # Verify paths exist
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory '{VIDEO_DIR}' not found!")
        print("Please create the directory and add your videos, or update VIDEO_DIR path")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please ensure your trained model is in the correct location")
        return
    
    # Check for timestamps file
    if os.path.exists(TIMESTAMPS_FILE):
        print(f"✓ Found timestamps file: {TIMESTAMPS_FILE}")
        print("Will use seizure-specific analysis for better precision")
    else:
        print(f"⚠ Timestamps file not found: {TIMESTAMPS_FILE}")
        print("Will process entire videos instead")
        TIMESTAMPS_FILE = None
    
    # Initialize inference engine
    print("\nInitializing seizure inference engine...")
    engine = SeizureInferenceEngine(MODEL_PATH)
    
    # Process all videos
    print(f"\nStarting batch inference on videos in: {VIDEO_DIR}")
    results = engine.process_video_directory(VIDEO_DIR, OUTPUT_DIR, TIMESTAMPS_FILE)
    
    print(f"\n🎉 Inference complete! Check '{OUTPUT_DIR}' for results.")
    
    # Additional timestamp-specific guidance
    if TIMESTAMPS_FILE:
        print("\n📊 TIMESTAMP-ENHANCED ANALYSIS:")
        print("✓ Analyzed only seizure segments (more precise than full videos)")
        print("✓ Individual seizure segment predictions available in detailed_results.json")
        print("✓ Multiple seizures per video handled automatically")
        print("✓ Summary shows both video-level and segment-level statistics")

if __name__ == "__main__":
    main()