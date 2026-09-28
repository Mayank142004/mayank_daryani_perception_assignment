import argparse
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist

from src.data.loader import NuScenesLoader
from src.utils.logging import logger, log_stage

def get_map_points(nusc_map, bounds, spacing=0.5):
    """
    Extract map lane divider and road divider points within bounds.
    bounds: (min_x, max_x, min_y, max_y)
    """
    pts = []
    for layer in ['lane_divider', 'road_divider']:
        if not hasattr(nusc_map, layer):
            continue
        for record in getattr(nusc_map, layer):
            line_nodes = [nusc_map.get('node', node_token) for node_token in record['node_tokens']]
            line_pts = np.array([[n['x'], n['y']] for n in line_nodes])
            
            # Simple bounding box check for the whole line
            if np.any((line_pts[:, 0] >= bounds[0]) & (line_pts[:, 0] <= bounds[1]) & 
                      (line_pts[:, 1] >= bounds[2]) & (line_pts[:, 1] <= bounds[3])):
                
                # Interpolate to uniform spacing
                for i in range(len(line_pts) - 1):
                    p1, p2 = line_pts[i], line_pts[i+1]
                    dist = np.linalg.norm(p1 - p2)
                    num_pts = max(2, int(dist / spacing))
                    # linspace
                    xs = np.linspace(p1[0], p2[0], num_pts)
                    ys = np.linspace(p1[1], p2[1], num_pts)
                    pts.append(np.column_stack((xs, ys)))
                    
    if pts:
        return np.vstack(pts)
    return np.zeros((0, 2))

def compute_chamfer(S1: np.ndarray, S2: np.ndarray) -> float:
    if len(S1) == 0 or len(S2) == 0:
        return float('inf')
        
    dists = cdist(S1, S2)
    # distance from S1 to S2
    d1 = np.mean(np.min(dists, axis=1))
    # distance from S2 to S1
    d2 = np.mean(np.min(dists, axis=0))
    
    return (d1 + d2) / 2.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()

    log_stage("Evaluate (Chamfer Distance)")
    
    loader = NuScenesLoader('config.yaml')
    
    try:
        scene_rec = next(s for s in loader.nusc.scene if s['name'] == args.scene)
    except StopIteration:
        logger.error(f"Scene {args.scene} not found.")
        return
        
    log_rec = loader.nusc.get('log', scene_rec['log_token'])
    map_name = log_rec['location']
    
    topology_path = Path('outputs') / f"topology_{args.scene}.json"
    if not topology_path.exists():
        logger.error(f"{topology_path} not found. Run topology extraction first.")
        return
        
    with open(topology_path, 'r') as f:
        graph = json.load(f)
        
    pred_pts = []
    for n in graph['nodes']:
        pred_pts.extend(n['polyline'])
    pred_pts = np.array(pred_pts)
    
    if len(pred_pts) == 0:
        logger.error("Topology graph has no nodes.")
        return

    # Try to load HD map
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
        nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name=map_name)
    except Exception as e:
        logger.warning(f"Could not load NuScenesMap for {map_name}. The HD map expansion may not be downloaded.")
        logger.warning(f"Exception: {e}")
        logger.info("Evaluation aborted due to missing ground truth map.")
        return
        
    # Bounding box of predictions + margin
    margin = 20.0
    min_x, max_x = np.min(pred_pts[:, 0]) - margin, np.max(pred_pts[:, 0]) + margin
    min_y, max_y = np.min(pred_pts[:, 1]) - margin, np.max(pred_pts[:, 1]) + margin
    bounds = (min_x, max_x, min_y, max_y)
    
    logger.info("Extracting HD map lane boundaries...")
    gt_pts = get_map_points(nusc_map, bounds)
    
    if len(gt_pts) == 0:
        logger.error("No ground truth lane boundaries found in map within prediction bounds.")
        return
        
    logger.info(f"Prediction points: {len(pred_pts)}, GT points: {len(gt_pts)}")
    
    logger.info("Computing Chamfer distance...")
    chamfer_dist = compute_chamfer(pred_pts, gt_pts)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Scene:             {args.scene}")
    print(f"Chamfer Distance:  {chamfer_dist:.4f} meters")
    print("="*40)

if __name__ == '__main__':
    main()
