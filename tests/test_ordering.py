import numpy as np
import yaml
from pathlib import Path
from src.vectorise.mask_to_lines import Vectoriser

def test_ordering(tmp_path: Path):
    # Setup dummy config
    config = {
        'vectorise': {
            'prob_threshold': 0.5,
            'use_drivable_gate': False,
            'drivable_dilate_kernel': 31,
            'min_pixel_count': 5,
            'min_vertical_extent': 5,
            'max_wh_ratio': 5.0
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f)
        
    vectoriser = Vectoriser(str(cfg_file))
    
    # Create a synthetic diagonal line
    ll_prob = np.zeros((100, 100), dtype=np.float32)
    da_mask = np.ones((100, 100), dtype=np.uint8)
    
    # Draw diagonal line from (10, 10) to (90, 90) -> top-left to bottom-right
    for i in range(10, 91):
        ll_prob[i, i] = 1.0
        
    polylines, rejects = vectoriser.process(da_mask, ll_prob)
    
    assert len(polylines) == 1
    points = polylines[0].points
    
    # Points should be ordered near-to-far (bottom of image to top)
    # v coordinate (row) should be monotonically decreasing
    v_coords = points[:, 1]
    
    # Check if monotonically decreasing
    diffs = np.diff(v_coords)
    assert np.all(diffs < 0), f"Rows are not monotonically decreasing: {v_coords}"
    
    # The first point should be (90, 90)
    assert np.allclose(points[0], [90, 90])
    # The last point should be (10, 10)
    assert np.allclose(points[-1], [10, 10])
