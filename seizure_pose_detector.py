import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SeizureDetector:
    """Real-time seizure detection using pose estimation and rule-based analysis"""
    
    def __init__(self, history_frames: int = 60, fps: int = 30):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection parameters
        self.history_frames = history_frames
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # Feature buffers
        self.keypoint_buffer = deque(maxlen=history_frames)
        self.movement_buffer = deque(maxlen=history_frames)
        self.symmetry_buffer = deque(maxlen=history_frames)
        self.jerk_buffer = deque(maxlen=history_frames)
        
        # Seizure detection thresholds (calibrated for common seizure patterns)
        self.thresholds = {
            'movement_spike': 0.15,      # Sudden large movements
            'repetitive_threshold': 0.7,  # Repetitive movement correlation
            'asymmetry_threshold': 0.3,   # Body asymmetry
            'jerk_threshold': 0.5,        # High jerk values
            'tremor_frequency': (3, 8),   # Hz range for tremors
        }
        
        # State tracking
        self.seizure_score = 0
        self.seizure_state = "Normal"
        self.state_history = deque(maxlen=150)  # 5 seconds at 30fps
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame and return seizure detection results"""
        # Extract pose
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {
                'state': 'No pose detected',
                'seizure_score': 0,
                'indicators': {},
                'annotated_frame': frame
            }
        
        # Extract keypoints
        keypoints = self._extract_keypoints(results.pose_landmarks)
        self.keypoint_buffer.append(keypoints)
        
        if len(self.keypoint_buffer) < 10:  # Need minimum frames
            return {
                'state': 'Initializing',
                'seizure_score': 0,
                'indicators': {},
                'annotated_frame': self._draw_pose(frame, results)
            }
        
        # Analyze features
        indicators = self._analyze_seizure_indicators()
        
        # Update seizure score
        self._update_seizure_score(indicators)
        
        # Determine state
        state = self._determine_state()
        
        # Annotate frame
        annotated_frame = self._draw_pose(frame, results)
        annotated_frame = self._add_status_overlay(annotated_frame, state, indicators)
        
        return {
            'state': state,
            'seizure_score': self.seizure_score,
            'indicators': indicators,
            'annotated_frame': annotated_frame
        }
    
    def _extract_keypoints(self, landmarks) -> np.ndarray:
        """Extract keypoint coordinates"""
        keypoints = []
        for landmark in landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints).reshape(-1, 4)
    
    def _analyze_seizure_indicators(self) -> Dict:
        """Analyze multiple indicators of seizure activity"""
        indicators = {}
        
        # 1. Movement magnitude analysis
        movement = self._analyze_movement()
        indicators['movement_intensity'] = movement['intensity']
        indicators['movement_spike'] = movement['spike_detected']
        
        # 2. Repetitive movement detection
        repetitive = self._detect_repetitive_movement()
        indicators['repetitive_movement'] = repetitive['score']
        indicators['dominant_frequency'] = repetitive['frequency']
        
        # 3. Body symmetry analysis
        symmetry = self._analyze_symmetry()
        indicators['asymmetry_score'] = symmetry['score']
        indicators['asymmetric_limbs'] = symmetry['affected_limbs']
        
        # 4. Jerk/acceleration analysis
        jerk = self._analyze_jerk()
        indicators['jerk_magnitude'] = jerk['magnitude']
        indicators['high_jerk_joints'] = jerk['affected_joints']
        
        # 5. Limb rigidity/extension
        rigidity = self._analyze_rigidity()
        indicators['rigidity_score'] = rigidity['score']
        indicators['extended_limbs'] = rigidity['extended_limbs']
        
        return indicators
    
    def _analyze_movement(self) -> Dict:
        """Analyze overall movement patterns"""
        if len(self.keypoint_buffer) < 2:
            return {'intensity': 0, 'spike_detected': False}
        
        # Calculate center of mass movement
        movements = []
        for i in range(1, len(self.keypoint_buffer)):
            prev_kp = self.keypoint_buffer[i-1]
            curr_kp = self.keypoint_buffer[i]
            
            # Use torso landmarks for center of mass
            torso_indices = [11, 12, 23, 24]  # Shoulders and hips
            prev_com = np.mean([prev_kp[idx][:2] for idx in torso_indices], axis=0)
            curr_com = np.mean([curr_kp[idx][:2] for idx in torso_indices], axis=0)
            
            movement = np.linalg.norm(curr_com - prev_com)
            movements.append(movement)
        
        movements = np.array(movements)
        self.movement_buffer.extend(movements)
        
        # Detect spikes
        if len(self.movement_buffer) > 10:
            recent_movement = list(self.movement_buffer)[-10:]
            spike_detected = max(recent_movement) > self.thresholds['movement_spike']
        else:
            spike_detected = False
        
        return {
            'intensity': float(np.mean(movements)),
            'spike_detected': spike_detected
        }
    
    def _detect_repetitive_movement(self) -> Dict:
        """Detect repetitive/rhythmic movements"""
        if len(self.keypoint_buffer) < 30:
            return {'score': 0, 'frequency': 0}
        
        # Analyze wrist movements (common in seizures)
        wrist_indices = [15, 16]  # Left and right wrists
        
        signals = []
        for wrist_idx in wrist_indices:
            positions = [kp[wrist_idx][:2] for kp in self.keypoint_buffer]
            signal = np.array(positions)
            
            # Calculate autocorrelation
            signal_centered = signal - np.mean(signal, axis=0)
            autocorr = np.correlate(signal_centered[:, 0], signal_centered[:, 0], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            signals.append(autocorr)
        
        # Find dominant frequency
        avg_autocorr = np.mean(signals, axis=0)
        
        # Look for peaks in autocorrelation
        peaks = []
        for i in range(1, len(avg_autocorr)-1):
            if avg_autocorr[i] > avg_autocorr[i-1] and avg_autocorr[i] > avg_autocorr[i+1] and avg_autocorr[i] > 0.3:
                peaks.append(i)
        
        if peaks:
            dominant_period = peaks[0]
            frequency = self.fps / dominant_period
            score = avg_autocorr[peaks[0]]
        else:
            frequency = 0
            score = 0
        
        return {
            'score': float(score),
            'frequency': float(frequency)
        }
    
    def _analyze_symmetry(self) -> Dict:
        """Analyze bilateral symmetry"""
        if not self.keypoint_buffer:
            return {'score': 0, 'affected_limbs': []}
        
        # Left-right pairs
        lr_pairs = {
            'arms': [(11, 12), (13, 14), (15, 16)],
            'legs': [(23, 24), (25, 26), (27, 28)]
        }
        
        asymmetry_scores = []
        affected_limbs = []
        
        for limb_type, pairs in lr_pairs.items():
            limb_asymmetry = []
            
            for left_idx, right_idx in pairs:
                # Compare positions over recent frames
                position_diffs = []
                for kp in list(self.keypoint_buffer)[-10:]:
                    left_pos = kp[left_idx][:3]
                    right_pos = kp[right_idx][:3]
                    
                    # Normalize by distance from center
                    center = (kp[11][:3] + kp[12][:3]) / 2  # Between shoulders
                    left_dist = np.linalg.norm(left_pos - center)
                    right_dist = np.linalg.norm(right_pos - center)
                    
                    diff = abs(left_dist - right_dist)
                    position_diffs.append(diff)
                
                limb_asymmetry.append(np.mean(position_diffs))
            
            avg_asymmetry = np.mean(limb_asymmetry)
            asymmetry_scores.append(avg_asymmetry)
            
            if avg_asymmetry > self.thresholds['asymmetry_threshold']:
                affected_limbs.append(limb_type)
        
        return {
            'score': float(np.mean(asymmetry_scores)),
            'affected_limbs': affected_limbs
        }
    
    def _analyze_jerk(self) -> Dict:
        """Analyze jerk (rate of change of acceleration)"""
        if len(self.keypoint_buffer) < 4:
            return {'magnitude': 0, 'affected_joints': []}
        
        # Focus on extremities
        joint_indices = {
            'left_wrist': 15,
            'right_wrist': 16,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        jerk_magnitudes = []
        affected_joints = []
        
        for joint_name, idx in joint_indices.items():
            positions = np.array([kp[idx][:3] for kp in list(self.keypoint_buffer)[-4:]])
            
            # Calculate derivatives
            velocity = np.diff(positions, axis=0)
            acceleration = np.diff(velocity, axis=0)
            jerk = np.diff(acceleration, axis=0)
            
            if len(jerk) > 0:
                jerk_mag = np.linalg.norm(jerk[0])
                jerk_magnitudes.append(jerk_mag)
                
                if jerk_mag > self.thresholds['jerk_threshold']:
                    affected_joints.append(joint_name)
        
        return {
            'magnitude': float(np.mean(jerk_magnitudes)) if jerk_magnitudes else 0,
            'affected_joints': affected_joints
        }
    
    def _analyze_rigidity(self) -> Dict:
        """Analyze limb rigidity and extension"""
        if not self.keypoint_buffer:
            return {'score': 0, 'extended_limbs': []}
        
        current_kp = self.keypoint_buffer[-1]
        
        # Check arm and leg extension
        limb_angles = {
            'left_arm': self._calculate_angle(current_kp[11][:3], current_kp[13][:3], current_kp[15][:3]),
            'right_arm': self._calculate_angle(current_kp[12][:3], current_kp[14][:3], current_kp[16][:3]),
            'left_leg': self._calculate_angle(current_kp[23][:3], current_kp[25][:3], current_kp[27][:3]),
            'right_leg': self._calculate_angle(current_kp[24][:3], current_kp[26][:3], current_kp[28][:3])
        }
        
        extended_limbs = []
        rigidity_scores = []
        
        for limb, angle in limb_angles.items():
            # Check if limb is extended (angle > 150 degrees)
            if angle > 150:
                extended_limbs.append(limb)
                rigidity_scores.append(1.0)
            else:
                rigidity_scores.append(0.0)
        
        return {
            'score': float(np.mean(rigidity_scores)),
            'extended_limbs': extended_limbs
        }
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)
    
    def _update_seizure_score(self, indicators: Dict):
        """Update overall seizure score based on indicators"""
        score = 0
        weights = {
            'movement_spike': 0.3,
            'repetitive_movement': 0.25,
            'asymmetry_score': 0.2,
            'jerk_magnitude': 0.15,
            'rigidity_score': 0.1
        }
        
        # Movement spike
        if indicators['movement_spike']:
            score += weights['movement_spike']
        
        # Repetitive movement
        if indicators['repetitive_movement'] > self.thresholds['repetitive_threshold']:
            freq = indicators['dominant_frequency']
            if self.thresholds['tremor_frequency'][0] <= freq <= self.thresholds['tremor_frequency'][1]:
                score += weights['repetitive_movement']
        
        # Asymmetry
        if indicators['asymmetry_score'] > self.thresholds['asymmetry_threshold']:
            score += weights['asymmetry_score']
        
        # High jerk
        if indicators['jerk_magnitude'] > self.thresholds['jerk_threshold']:
            score += weights['jerk_magnitude']
        
        # Rigidity
        if indicators['rigidity_score'] > 0.5:
            score += weights['rigidity_score']
        
        # Smooth score updates
        self.seizure_score = 0.8 * self.seizure_score + 0.2 * score
        self.state_history.append(self.seizure_score)
    
    def _determine_state(self) -> str:
        """Determine current seizure state"""
        if self.seizure_score < 0.2:
            return "Normal"
        elif self.seizure_score < 0.4:
            return "Mild abnormal movement"
        elif self.seizure_score < 0.6:
            return "Potential pre-seizure"
        elif self.seizure_score < 0.8:
            return "High seizure risk"
        else:
            return "SEIZURE DETECTED"
    
    def _draw_pose(self, image: np.ndarray, results) -> np.ndarray:
        """Draw pose landmarks on image"""
        annotated = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2
            )
        )
        return annotated
    
    def _add_status_overlay(self, image: np.ndarray, state: str, indicators: Dict) -> np.ndarray:
        """Add status information overlay"""
        height, width = image.shape[:2]
        
        # Determine color based on state
        if "SEIZURE" in state:
            color = (0, 0, 255)  # Red
        elif "risk" in state or "pre-seizure" in state:
            color = (0, 165, 255)  # Orange
        elif "abnormal" in state:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green
        
        # Main status
        cv2.putText(image, f"Status: {state}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Seizure score bar
        bar_width = int(200 * self.seizure_score)
        cv2.rectangle(image, (10, 50), (210, 70), (255, 255, 255), 2)
        cv2.rectangle(image, (10, 50), (10 + bar_width, 70), color, -1)
        cv2.putText(image, f"Risk: {self.seizure_score:.2f}", (220, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Key indicators
        y_offset = 100
        if indicators.get('movement_spike'):
            cv2.putText(image, "! Sudden movement detected", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
        
        if indicators.get('repetitive_movement', 0) > 0.7:
            cv2.putText(image, f"! Repetitive movement ({indicators['dominant_frequency']:.1f} Hz)",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
        
        if indicators.get('asymmetric_limbs'):
            cv2.putText(image, f"! Asymmetry: {', '.join(indicators['asymmetric_limbs'])}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_offset += 20
        
        if indicators.get('high_jerk_joints'):
            cv2.putText(image, f"! High jerk: {', '.join(indicators['high_jerk_joints'])}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_offset += 20
        
        return image


def main():
    """Run seizure detection on webcam or video file"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time seizure detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save output video')
    args = parser.parse_args()
    
    # Initialize detector
    detector = SeizureDetector()
    
    # Open video source
    if args.source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer if saving
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
    
    print("Starting seizure detection...")
    print("Press 'q' to quit, 's' to simulate seizure-like movement")
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Display results
        cv2.imshow('Seizure Detection', result['annotated_frame'])
        
        # Save if requested
        if args.save:
            out.write(result['annotated_frame'])
        
        # Print alerts
        if "SEIZURE" in result['state'] and frame_count % 30 == 0:
            print(f"\n⚠️ ALERT: {result['state']}")
            print(f"Indicators: {result['indicators']}")
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Simulating seizure-like movement...")
            # This would trigger test movements in a real implementation
        
        frame_count += 1
        
        # FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed
            print(f"\rFPS: {actual_fps:.1f} | State: {result['state']}", end='')
    
    # Cleanup
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()
    
    print("\nDetection completed.")


if __name__ == "__main__":
    main()
    