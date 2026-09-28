import numpy as np
import yaml
from pathlib import Path
from src.geometry.ipm import IPMProjector

def test_ipm_roundtrip(tmp_path: Path):
    config = {
        'project': {
            'clip_x': [0.0, 50.0],
            'clip_y': [-15.0, 15.0],
            'confidence_decay_rate': 0.02
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f)
        
    projector = IPMProjector(str(cfg_file))
    
    # Dummy calibration
    # nuScenes CAM_FRONT roughly
    K = np.array([
        [1266.0, 0.0, 800.0],
        [0.0, 1266.0, 450.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Simple forward facing camera, mounted 1.5m high, looking slightly down (pitch)
    # Let's just use identity rotation (looking forward along Z, X right, Y down)
    # Wait, NuScenes ego frame: X forward, Y left, Z up
    # NuScenes cam frame: X right, Y down, Z forward
    # R maps cam -> ego
    # X_ego = Z_cam
    # Y_ego = -X_cam
    # Z_ego = -Y_cam
    R_cam_to_ego = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=np.float64)
    
    t_cam_to_ego = np.array([1.0, 0.0, 1.5]) # 1m forward, 1.5m high
    
    # Define a pixel point (bottom center of image)
    u_orig, v_orig = 800.0, 800.0
    
    p_ego = projector.project_point(u_orig, v_orig, K, R_cam_to_ego, t_cam_to_ego)
    
    assert p_ego is not None
    assert p_ego[2] == 0.0  # Must be on ground plane
    
    # Unproject
    u_unproj, v_unproj = projector.unproject_point(p_ego, K, R_cam_to_ego, t_cam_to_ego)
    
    error = np.linalg.norm([u_orig - u_unproj, v_orig - v_unproj])
    assert error < 2.0, f"Max error {error} exceeded 2px threshold."
