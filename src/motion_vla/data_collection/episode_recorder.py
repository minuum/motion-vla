"""Episode recorder for real robot data collection.

Records demonstrations with images, robot states, and actions for π0 fine-tuning.
"""

import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2


class EpisodeRecorder:
    """Records robot demonstration episodes in HDF5 format."""
    
    def __init__(self, task_name: str, data_dir: str = "data"):
        """Initialize episode recorder.
        
        Args:
            task_name: Name of task (e.g., 'pushing', 'pick_place')
            data_dir: Root directory for saving episodes
        """
        self.task = task_name
        self.data_dir = Path(data_dir) / task_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_id = self._get_next_episode_id()
        
        # Recording buffers
        self.images: List[np.ndarray] = []
        self.robot_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.recording = False
        
    def _get_next_episode_id(self) -> int:
        """Get next available episode ID."""
        existing = list(self.data_dir.glob("episode_*.h5"))
        if not existing:
            return 0
        ids = [int(p.stem.split('_')[1]) for p in existing]
        return max(ids) + 1
    
    def start_recording(self, language_instruction: str):
        """Start recording a new episode.
        
        Args:
            language_instruction: Natural language task description
        """
        self.language = language_instruction
        self.images = []
        self.robot_states = []
        self.actions = []
        self.recording = True
        self.start_time = datetime.now()
        print(f"Started recording episode {self.episode_id}: {language_instruction}")
    
    def add_step(self, 
                 image: np.ndarray,
                 robot_state: np.ndarray,
                 action: np.ndarray):
        """Add a timestep to current episode.
        
        Args:
            image: RGB image (H, W, 3)
            robot_state: Robot joint positions (6,) or (7,)
            action: Action taken (6,) for pushing or (7,) for pick&place
        """
        if not self.recording:
            raise RuntimeError("Not recording. Call start_recording() first.")
        
        self.images.append(image)
        self.robot_states.append(robot_state)
        self.actions.append(action)
    
    def stop_recording(self, success: bool = True, notes: str = ""):
        """Stop recording and save episode.
        
        Args:
            success: Whether episode was successful
            notes: Optional notes about the episode
        
        Returns:
            Path to saved HDF5 file
        """
        if not self.recording:
            raise RuntimeError("Not recording.")
        
        self.recording = False
        
        # Convert to numpy arrays
        images = np.stack(self.images, axis=0)  # (T, H, W, 3)
        robot_states = np.stack(self.robot_states, axis=0)  # (T, 6 or 7)
        actions = np.stack(self.actions, axis=0)  # (T, 6 or 7)
        
        # Save to HDF5
        filename = self.data_dir / f"episode_{self.episode_id:04d}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs['task'] = self.task
            f.attrs['language'] = self.language
            f.attrs['timestamp'] = self.start_time.isoformat()
            f.attrs['duration'] = (datetime.now() - self.start_time).total_seconds()
            f.attrs['robot'] = 'Dobot_E6'
            f.attrs['success'] = success
            f.attrs['notes'] = notes
            f.attrs['num_steps'] = len(images)
            
            # Observations
            obs_group = f.create_group('observations')
            obs_group.create_dataset('images', data=images, compression='gzip')
            obs_group.create_dataset('robot_state', data=robot_states)
            
            # Actions
            f.create_dataset('actions', data=actions)
        
        print(f"Saved episode {self.episode_id} ({len(images)} steps, success={success})")
        print(f"  → {filename}")
        
        self.episode_id += 1
        return filename
    
    def get_statistics(self) -> Dict:
        """Get statistics of collected data."""
        episodes = list(self.data_dir.glob("episode_*.h5"))
        
        if not episodes:
            return {"num_episodes": 0}
        
        total_steps = 0
        success_count = 0
        
        for ep_file in episodes:
            with h5py.File(ep_file, 'r') as f:
                total_steps += f.attrs['num_steps']
                if f.attrs['success']:
                    success_count += 1
        
        return {
            "num_episodes": len(episodes),
            "success_rate": success_count / len(episodes),
            "total_steps": total_steps,
            "avg_steps": total_steps / len(episodes)
        }


def visualize_episode(episode_path: str, output_video: Optional[str] = None):
    """Visualize an episode as a video.
    
    Args:
        episode_path: Path to episode HDF5 file
        output_video: Optional path to save video file
    """
    with h5py.File(episode_path, 'r') as f:
        images = f['observations/images'][:]
        language = f.attrs['language']
        success = f.attrs['success']
        
        print(f"Episode: {language}")
        print(f"Success: {success}")
        print(f"Steps: {len(images)}")
        
        if output_video:
            # Save as video
            h, w = images[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, 10.0, (w, h))
            
            for img in images:
                # Add text overlay
                img_copy = img.copy()
                cv2.putText(img_copy, language, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f"Saved video: {output_video}")
        else:
            # Display interactively
            for img in images:
                cv2.imshow('Episode', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test recorder
    recorder = EpisodeRecorder("pushing")
    
    recorder.start_recording("Push the block to the left")
    
    # Simulate 50 timesteps
    for i in range(50):
        fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        fake_state = np.random.randn(6).astype(np.float32)
        fake_action = np.random.randn(6).astype(np.float32)
        
        recorder.add_step(fake_image, fake_state, fake_action)
    
    filepath = recorder.stop_recording(success=True)
    
    # Show statistics
    stats = recorder.get_statistics()
    print(f"\nDataset statistics: {stats}")
