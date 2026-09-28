from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import yaml
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from src.utils.logging import logger

@dataclass
class Frame:
    token: str
    timestamp: int
    image_path: str
    camera_matrix: np.ndarray # 3x3
    cam_to_ego_t: np.ndarray # 3,
    cam_to_ego_R: np.ndarray # 3x3
    ego_to_global_t: np.ndarray # 3,
    ego_to_global_R: np.ndarray # 3x3

    def ego_to_global(self, points_ego: np.ndarray) -> np.ndarray:
        """
        Transform points from ego frame to global frame.
        points_ego: (N, 3)
        """
        return (self.ego_to_global_R @ points_ego.T).T + self.ego_to_global_t

    def global_to_ego(self, points_global: np.ndarray) -> np.ndarray:
        """
        Transform points from global frame to ego frame.
        points_global: (N, 3)
        """
        return (self.ego_to_global_R.T @ (points_global - self.ego_to_global_t).T).T

class NuScenesLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']
        
        logger.info(f"Loading NuScenes version {self.config['version']} from {self.config['dataroot']}")
        self.nusc = NuScenes(version=self.config['version'], dataroot=self.config['dataroot'], verbose=False)
        logger.info("NuScenes loaded successfully.")

    def list_scenes(self) -> List[Dict[str, str]]:
        return [{'name': s['name'], 'description': s['description'], 'token': s['token']} for s in self.nusc.scene]

    def load_scene(self, scene_name: str, max_frames: Optional[int] = None) -> List[Frame]:
        try:
            scene = next(s for s in self.nusc.scene if s['name'] == scene_name)
        except StopIteration:
            raise ValueError(f"Scene {scene_name} not found in dataset.")

        current_token = scene['first_sample_token']
        frames = []

        logger.info(f"Extracting keyframes for {scene_name}...")
        while current_token:
            if max_frames and len(frames) >= max_frames:
                break
            
            sample = self.nusc.get('sample', current_token)
            cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            
            # Intrinsics
            calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            K = np.array(calib['camera_intrinsic'])
            
            # Extrinsics: sensor to ego
            cam_to_ego_t = np.array(calib['translation'])
            # Quaternions converted to rotation matrices ONCE at load time
            cam_to_ego_R = Quaternion(calib['rotation']).rotation_matrix

            # Pose: ego to global
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_to_global_t = np.array(ego_pose['translation'])
            ego_to_global_R = Quaternion(ego_pose['rotation']).rotation_matrix

            img_path = self.nusc.get_sample_data_path(cam_data['token'])

            frames.append(Frame(
                token=sample['token'],
                timestamp=sample['timestamp'],
                image_path=str(img_path),
                camera_matrix=K,
                cam_to_ego_t=cam_to_ego_t,
                cam_to_ego_R=cam_to_ego_R,
                ego_to_global_t=ego_to_global_t,
                ego_to_global_R=ego_to_global_R
            ))
            current_token = sample['next']
            
        logger.info(f"Loaded {len(frames)} keyframes.")
        return frames
