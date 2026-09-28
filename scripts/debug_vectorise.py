import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Debug Vectorise")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser('config.yaml')
    
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"vectorise_{args.scene}.mp4"
    
    first_img = cv2.imread(frames[0].image_path)
    h, w = first_img.shape[:2]
    fps = 2.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    logger.info(f"Generating video {out_path}...")
    
    # Pre-generate some distinct colours
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3)).tolist()
    
    for i, frame in enumerate(tqdm(frames)):
        img = cv2.imread(frame.image_path)
        res = segmenter.segment(frame)
        
        polylines, rejects = vectoriser.process(res.drivable_mask, res.lane_mask)
        
        # Report
        pts_per_pl = [len(pl.points) for pl in polylines]
        print(f"\nFrame {i+1}: Found {len(polylines)} components.")
        print(f"Rejections: {rejects}")
        print(f"Points per polyline: {pts_per_pl}")
        
        # Draw
        for j, pl in enumerate(polylines):
            color = colors[j % len(colors)]
            pts = pl.points.astype(np.int32)
            for k in range(len(pts) - 1):
                cv2.line(img, tuple(pts[k]), tuple(pts[k+1]), color, 2)
            
            # Draw start point thicker to verify ordering (near -> far)
            if len(pts) > 0:
                cv2.circle(img, tuple(pts[0]), 5, (0, 255, 255), -1) # Yellow dot at start (bottom of image)
                
        writer.write(img.astype(np.uint8))
        
    writer.release()
    logger.info("Done.")

if __name__ == '__main__':
    main()
