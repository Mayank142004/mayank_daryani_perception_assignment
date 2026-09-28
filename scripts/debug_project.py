import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.geometry.ipm import IPMProjector
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Debug Project (IPM)")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser('config.yaml')
    projector = IPMProjector('config.yaml')
    
    out_dir = Path('outputs') / f"ipm_debug_{args.scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating BEV scatter plots in {out_dir}...")
    
    for i, frame in enumerate(tqdm(frames)):
        # 1. Segment
        res = segmenter.segment(frame)
        
        # 2. Vectorise
        polylines, _ = vectoriser.process(res.drivable_mask, res.lane_mask)
        
        # 3. Project
        projected = projector.project(
            polylines, 
            frame.camera_matrix, 
            frame.cam_to_ego_R, 
            frame.cam_to_ego_t
        )
        
        # Plot BEV
        plt.figure(figsize=(6, 10))
        
        # Plot each polyline
        for pl in projected:
            pts = pl.points
            # Confidences mapping to alpha
            alphas = np.clip(pl.confidences, 0.1, 1.0)
            
            # Plot lines
            plt.plot(pts[:, 1], pts[:, 0], color='blue', alpha=0.5, linewidth=1)
            # Plot points
            plt.scatter(pts[:, 1], pts[:, 0], c=alphas, cmap='viridis', s=10, vmin=0, vmax=1)
            
        plt.plot(0, 0, marker='^', color='red', markersize=10, label='Ego')
        
        plt.xlim(15, -15) # Y is left in NuScenes ego frame (so left=positive, plot it inverted visually to match BEV)
        plt.ylim(0, 50)   # X is forward
        
        plt.xlabel("Left <--- Ego Y (m) ---> Right")
        plt.ylabel("Ego X (m) Forward")
        plt.title(f"BEV IPM - Frame {i+1}")
        plt.grid(True)
        
        out_path = out_dir / f"{i:03d}.png"
        plt.savefig(out_path, dpi=100)
        plt.close()
        
    logger.info("Done.")

if __name__ == '__main__':
    main()
