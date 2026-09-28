import numpy as np
import yaml
from pathlib import Path
from src.stitch.temporal import Tracker, Track, local_heading

def test_stitch(tmp_path: Path):
    config = {
        'stitch': {
            'heading_n_points': 5,
            'gate_dist_m': 1.5,
            'gate_angle_deg': 45.0,
            'cost_angle_weight': 1.0,
            'bin_spacing_m': 0.5,
            'savgol_polyorder': 2,
            'max_misses': 3,
            'duplicate_gate_dist_m': 1.0,
            'duplicate_gate_angle_deg': 15.0
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f)
        
    tracker = Tracker(str(cfg_file))
    
    # 1. Arc-length fusion on curved track
    # Generate a noisy curve
    t = np.linspace(0, np.pi/2, 50)
    x = 10 * np.cos(t)
    y = 10 * np.sin(t)
    noise_pts1 = np.column_stack((x, y)) + np.random.normal(0, 0.2, (50, 2))
    noise_pts2 = np.column_stack((x, y)) + np.random.normal(0, 0.2, (50, 2))
    
    trk = Track(points=noise_pts1, confidences=np.ones(50))
    tracker.fuse_detection(trk, noise_pts2, np.ones(50))
    
    # Assert mean turn per point < 5 deg
    pts = trk.points
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    turns = np.abs(np.diff(angles))
    # handle 2pi wrap
    turns = np.where(turns > np.pi, 2*np.pi - turns, turns)
    mean_turn = np.rad2deg(np.mean(turns))
    
    assert mean_turn < 5.0, f"Mean turn per point was {mean_turn:.2f} deg, > 5"
    
    # 2. Hungarian association
    trk1 = Track(id="A", points=np.array([[0,0], [1,0], [2,0]]), confidences=np.ones(3))
    trk2 = Track(id="B", points=np.array([[0,5], [1,5], [2,5]]), confidences=np.ones(3))
    tracker.live_tracks = [trk1, trk2]
    
    det1 = np.array([[2.1, 0.1], [3.1, 0.1]]) # Should match A
    det2 = np.array([[2.1, 4.9], [3.1, 4.9]]) # Should match B
    
    matches, unmatched_trk, unmatched_det = tracker.match([det1, det2])
    
    assert len(matches) == 2
    assert (0, 0) in matches or (0, 1) in matches
    
    # Check if they match the correct ones (A to det1, B to det2)
    m_dict = dict(matches)
    assert m_dict[0] == 0 # A matches det1
    assert m_dict[1] == 1 # B matches det2
