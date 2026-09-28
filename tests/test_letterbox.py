import numpy as np
from src.segment.letterbox import letterbox, unletterbox_points

def test_letterbox_roundtrip():
    # Original image shape (e.g. nuScenes 1600x900)
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    
    # Letterbox to YOLOPv2 expected size (e.g. 640 width, 384 height)
    lb_img, r, pad = letterbox(img, new_shape=(384, 640))
    
    # Define some original points (x, y)
    original_pts = np.array([
        [0, 0],
        [1599, 899],
        [800, 450],
        [123, 456]
    ], dtype=np.float32)
    
    # Map original to letterbox manually
    lb_pts = original_pts * r + np.array(pad)
    
    # Run inverse mapping
    unmapped = unletterbox_points(lb_pts, r, pad)
    
    # Error should be essentially zero (floating point)
    error = np.linalg.norm(original_pts - unmapped, axis=1)
    
    assert np.all(error < 2.0), f"Max error {np.max(error)} exceeded 2px threshold."
