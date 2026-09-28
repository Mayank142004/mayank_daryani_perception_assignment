import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.geometry.ipm import IPMProjector
from src.stitch.temporal import Tracker
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Debug Stitch")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser('config.yaml')
    projector = IPMProjector('config.yaml')
    tracker = Tracker('config.yaml')
    
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"stitch_bev_{args.scene}.png"
    
    logger.info("Processing frames for stitching...")
    
    ego_traj = []
    
    for i, frame in enumerate(tqdm(frames)):
        ego_traj.append(frame.ego_to_global_t[:2])
        
        # 1. Segment
        res = segmenter.segment(frame)
        
        # 2. Vectorise
        polylines, _ = vectoriser.process(res.drivable_mask, res.lane_mask)
        
        # 3. Project to ego
        projected = projector.project(
            polylines, 
            frame.camera_matrix, 
            frame.cam_to_ego_R, 
            frame.cam_to_ego_t
        )
        
        # 4. Ego to Global
        global_detections = []
        for pl in projected:
            pts_ego_3d = np.column_stack((pl.points, np.zeros(len(pl.points))))
            pts_global_3d = frame.ego_to_global(pts_ego_3d)
            # update projected points to global xy
            pl.points = pts_global_3d[:, :2]
            global_detections.append(pl)
            
        # 5. Track update
        active, matches, new_trks, archived = tracker.update(global_detections)
        
        print(f"\nFrame {i+1}: Active={active}, Matches={matches}, New={new_trks}, Archived={archived}")
        
    logger.info("Running duplicate fusion pass...")
    tracker.duplicate_fusion()
    
    all_tracks = tracker.get_all_tracks()
    logger.info(f"Total tracks after duplicate fusion: {len(all_tracks)}")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    
    # Ego traj
    ego_traj = np.array(ego_traj)
    plt.plot(ego_traj[:, 0], ego_traj[:, 1], color='black', linewidth=2, label='Ego Path')
    
    # Tracks
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / max(1, len(all_tracks))) for i in range(len(all_tracks))]
    for i, trk in enumerate(all_tracks):
        pts = trk.points
        if len(pts) > 0:
            plt.plot(pts[:, 0], pts[:, 1], color=colors[i], linewidth=2)
            
    plt.title(f"Cumulative Global BEV Tracks - {args.scene}")
    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved stitch plot to {out_path}")

if __name__ == '__main__':
    main()
