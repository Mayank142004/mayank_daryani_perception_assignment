import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Tuple
from src.vectorise.mask_to_lines import Polyline

@dataclass
class ProjectedPolyline:
    points: np.ndarray      # (N, 2) [x, y] in ego frame (metres)
    confidences: np.ndarray # (N,) decayed confidences

class IPMProjector:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['project']
            
    def project(self, polylines: List[Polyline], K: np.ndarray, 
                R_cam_to_ego: np.ndarray, t_cam_to_ego: np.ndarray) -> List[ProjectedPolyline]:
        
        inv_K = np.linalg.inv(K)
        projected = []
        
        x_min, x_max = self.config['clip_x']
        y_min, y_max = self.config['clip_y']
        decay_rate = self.config['confidence_decay_rate']
        
        for pl in polylines:
            pts_ego = []
            confs = []
            
            for i, (u, v) in enumerate(pl.points):
                # 1. Pixel to camera ray
                ray_cam = inv_K @ np.array([u, v, 1.0])
                
                # 2. Camera ray to ego frame
                ray_ego = R_cam_to_ego @ ray_cam
                
                # 3. Intersect with ground plane z = 0
                # t_ego[2] is camera height. If ray points up or horizontal, ignore.
                if ray_ego[2] >= -1e-6:
                    continue
                    
                s = -t_cam_to_ego[2] / ray_ego[2]
                
                # If behind camera, ignore
                if s <= 0:
                    continue
                    
                p_ego = t_cam_to_ego + s * ray_ego
                
                x, y = p_ego[0], p_ego[1]
                
                # Clip to bounds
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    pts_ego.append([x, y])
                    
                    # Confidence decay based on range
                    r = np.sqrt(x**2 + y**2)
                    decay = np.exp(-decay_rate * r)
                    confs.append(pl.confidences[i] * decay)
                    
            if len(pts_ego) >= 2: # Keep polylines with at least 2 points
                projected.append(ProjectedPolyline(
                    points=np.array(pts_ego),
                    confidences=np.array(confs)
                ))
                
        return projected

    def project_point(self, u: float, v: float, K: np.ndarray, 
                      R_cam_to_ego: np.ndarray, t_cam_to_ego: np.ndarray) -> np.ndarray:
        """Utility for a single point, without bounds/decay, for testing."""
        inv_K = np.linalg.inv(K)
        ray_cam = inv_K @ np.array([u, v, 1.0])
        ray_ego = R_cam_to_ego @ ray_cam
        if ray_ego[2] >= -1e-6:
            return None
        s = -t_cam_to_ego[2] / ray_ego[2]
        return t_cam_to_ego + s * ray_ego

    def unproject_point(self, p_ego: np.ndarray, K: np.ndarray, 
                        R_cam_to_ego: np.ndarray, t_cam_to_ego: np.ndarray) -> np.ndarray:
        """Inverse projection from ground plane to pixel."""
        # p_ego = t + s * R @ d_cam
        # s * R @ d_cam = p_ego - t
        p_rel = p_ego - t_cam_to_ego
        ray_cam = np.linalg.inv(R_cam_to_ego) @ p_rel
        # ray_cam is scaled by s. Projecting via K normalizes it automatically
        # because we divide by Z.
        p_img = K @ ray_cam
        return np.array([p_img[0] / p_img[2], p_img[1] / p_img[2]])
