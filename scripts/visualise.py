import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

from src.data.loader import NuScenesLoader
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Visualise Reprojection")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        return
        
    topology_path = Path('outputs') / f"topology_{args.scene}.json"
    if not topology_path.exists():
        logger.error(f"{topology_path} not found. Run topology extraction first.")
        return
        
    with open(topology_path, 'r') as f:
        graph = json.load(f)
        
    out_dir = Path('outputs')
    out_path = out_dir / f"reprojection_{args.scene}.mp4"
    
    first_img = cv2.imread(frames[0].image_path)
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 2.0, (w, h))

    logger.info(f"Generating reprojection video {out_path}...")
    
    # Pre-parse nodes
    nodes = []
    for n in graph['nodes']:
        pts = np.array(n['polyline'])
        nodes.append(pts)

    for frame in tqdm(frames):
        img = cv2.imread(frame.image_path)
        K = frame.camera_matrix
        
        for pts_2d_global in nodes:
            # Recover global Z assuming points lie on the current ego frame's Z=0 plane
            # P_ego = R^T * (P_global - t)
            # We want Z_ego = 0.
            # Let r3 = third row of R^T (which is the third column of R)
            t_eg = frame.ego_to_global_t
            r3 = frame.ego_to_global_R[:, 2] # third column
            
            x_g = pts_2d_global[:, 0]
            y_g = pts_2d_global[:, 1]
            
            # r3[0]*(x - tx) + r3[1]*(y - ty) + r3[2]*(z - tz) = 0
            # z = tz - (r3[0]*(x - tx) + r3[1]*(y - ty)) / r3[2]
            z_g = t_eg[2] - (r3[0] * (x_g - t_eg[0]) + r3[1] * (y_g - t_eg[1])) / r3[2]
            
            pts_3d = np.column_stack((x_g, y_g, z_g))
            
            # Global to Ego
            pts_ego = frame.global_to_ego(pts_3d)
            
            # Ego to Cam
            p_cam = (frame.cam_to_ego_R.T @ (pts_ego - frame.cam_to_ego_t).T).T
            
            # Keep points in front of camera (Z > 0 in cam frame)
            mask = p_cam[:, 2] > 0.1
            if not np.any(mask):
                continue
                
            p_cam = p_cam[mask]
            
            # Project to image
            p_img = (K @ p_cam.T).T
            u = p_img[:, 0] / p_img[:, 2]
            v = p_img[:, 1] / p_img[:, 2]
            
            pts_2d = np.column_stack((u, v)).astype(np.int32)
            
            # Draw lines
            for k in range(len(pts_2d) - 1):
                p1, p2 = pts_2d[k], pts_2d[k+1]
                # Filter out crazy lines that cross the screen behind
                if np.linalg.norm(p1 - p2) > 1000:
                    continue
                cv2.line(img, tuple(p1), tuple(p2), (0, 255, 255), 2)
                
        writer.write(img)
        
    writer.release()
    logger.info("Done.")

if __name__ == '__main__':
    main()
