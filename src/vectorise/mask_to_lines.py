import yaml
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label
from dataclasses import dataclass
from typing import List, Dict, Tuple
from src.utils.logging import logger

@dataclass
class Polyline:
    points: np.ndarray  # (N, 2) [u, v]
    confidences: np.ndarray # (N,)

class Vectoriser:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['vectorise']
            
    def process(self, da_mask: np.ndarray, ll_prob: np.ndarray) -> Tuple[List[Polyline], Dict[str, int]]:
        cfg = self.config
        
        # 1. Threshold
        ll_bin = ll_prob >= cfg['prob_threshold']
        
        # 2. Gate with dilated drivable area
        if cfg['use_drivable_gate']:
            k_size = cfg['drivable_dilate_kernel']
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            da_dilated = cv2.dilate(da_mask.astype(np.uint8), kernel)
            ll_bin = ll_bin & (da_dilated > 0)
            
        # 3. Skeletonise
        skeleton = skeletonize(ll_bin)
        
        # 4. Label connected components
        labeled, num_components = label(skeleton, return_num=True)
        
        polylines = []
        reject_counts = {
            'min_pixels': 0,
            'min_vert_extent': 0,
            'max_wh_ratio': 0
        }
        
        # 5 & 6. Filtering and ordering
        for i in range(1, num_components + 1):
            mask_i = (labeled == i)
            # Find coordinates of True values (rows, cols) -> (v, u)
            coords = np.argwhere(mask_i)
            if len(coords) == 0:
                continue
                
            v = coords[:, 0]
            u = coords[:, 1]
            
            # Rejection filters
            if len(coords) < cfg['min_pixel_count']:
                reject_counts['min_pixels'] += 1
                continue
                
            v_extent = v.max() - v.min()
            if v_extent < cfg['min_vertical_extent']:
                reject_counts['min_vert_extent'] += 1
                continue
                
            u_extent = u.max() - u.min()
            # Avoid division by zero
            wh_ratio = u_extent / max(1, v_extent)
            if wh_ratio > cfg['max_wh_ratio']:
                reject_counts['max_wh_ratio'] += 1
                continue
                
            # 6. Order NEAR-TO-FAR (bottom of image to top)
            # Image rows: larger v is near (bottom), smaller v is far (top)
            # So sort by v descending
            # Group by row, take median col
            unique_v = np.unique(v)
            unique_v = np.sort(unique_v)[::-1]
            
            ordered_points = []
            ordered_confs = []
            
            for row in unique_v:
                cols = u[v == row]
                med_col = np.median(cols)
                ordered_points.append([med_col, row])
                # 7. confidence from mask probability
                conf = ll_prob[int(row), int(med_col)]
                ordered_confs.append(conf)
                
            polylines.append(Polyline(
                points=np.array(ordered_points),
                confidences=np.array(ordered_confs)
            ))
            
        return polylines, reject_counts
