import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Debug Masks")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"masks_{args.scene}.mp4"
    
    # Setup VideoWriter based on first frame
    first_img = cv2.imread(frames[0].image_path)
    h, w = first_img.shape[:2]
    fps = 2.0  # NuScenes keyframes are 2 Hz
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    logger.info(f"Generating video {out_path}...")
    
    for i, frame in enumerate(tqdm(frames)):
        img = cv2.imread(frame.image_path)
        res = segmenter.segment(frame)
        
        # Overlay Drivable Area (Green)
        green_mask = (res.drivable_mask == 1)
        img[green_mask] = img[green_mask] * 0.5 + np.array([0, 255, 0], dtype=np.float64) * 0.5
        
        # Overlay Lane Lines (Red)
        red_mask = (res.lane_mask == 1)
        img[red_mask] = img[red_mask] * 0.5 + np.array([0, 0, 255], dtype=np.float64) * 0.5
        
        writer.write(img.astype(np.uint8))
        
    writer.release()
    logger.info("Done.")

if __name__ == '__main__':
    main()
