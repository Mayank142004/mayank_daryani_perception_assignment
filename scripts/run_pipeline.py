import argparse
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.geometry.ipm import IPMProjector
from src.stitch.temporal import Tracker
from src.graph.topology import TopologyBuilder
from src.graph.schema import validate_and_save
from src.utils.logging import logger, log_stage

def generate_bev_plot(graph, frames, out_path, scene_name):
    logger.info(f"Generating BEV plot: {out_path}")
    plt.figure(figsize=(10, 10))
    
    # Plot ego trajectory
    ego_traj = []
    for f in frames:
        ego_traj.append(f.ego_to_global_t[:2])
    ego_traj = np.array(ego_traj)
    if len(ego_traj) > 0:
        plt.plot(ego_traj[:, 0], ego_traj[:, 1], color='black', linewidth=2, label='Ego Path')
        
    # Plot graph nodes
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / max(1, len(graph.nodes))) for i in range(len(graph.nodes))]
    
    for i, n in enumerate(graph.nodes):
        pts = np.array(n.polyline)
        if len(pts) > 0:
            plt.plot(pts[:, 0], pts[:, 1], color=colors[i], linewidth=2)
            
    plt.title(f"Topology Graph BEV - {scene_name}")
    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_reprojection(graph, frames, out_path):
    logger.info(f"Generating reprojection video {out_path}...")
    first_img = cv2.imread(frames[0].image_path)
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 2.0, (w, h))

    nodes = []
    for n in graph.nodes:
        nodes.append(np.array(n.polyline))

    for frame in tqdm(frames):
        img = cv2.imread(frame.image_path)
        K = frame.camera_matrix
        
        for pts_2d_global in nodes:
            t_eg = frame.ego_to_global_t
            r3 = frame.ego_to_global_R[:, 2] 
            
            x_g = pts_2d_global[:, 0]
            y_g = pts_2d_global[:, 1]
            z_g = t_eg[2] - (r3[0] * (x_g - t_eg[0]) + r3[1] * (y_g - t_eg[1])) / r3[2]
            
            pts_3d = np.column_stack((x_g, y_g, z_g))
            pts_ego = frame.global_to_ego(pts_3d)
            p_cam = (frame.cam_to_ego_R.T @ (pts_ego - frame.cam_to_ego_t).T).T
            
            mask = p_cam[:, 2] > 0.1
            if not np.any(mask):
                continue
                
            p_cam = p_cam[mask]
            p_img = (K @ p_cam.T).T
            u = p_img[:, 0] / p_img[:, 2]
            v = p_img[:, 1] / p_img[:, 2]
            
            pts_2d = np.column_stack((u, v)).astype(np.int32)
            for k in range(len(pts_2d) - 1):
                p1, p2 = pts_2d[k], pts_2d[k+1]
                if np.linalg.norm(p1 - p2) > 1000:
                    continue
                cv2.line(img, tuple(p1), tuple(p2), (0, 255, 255), 2)
                
        writer.write(img)
    writer.release()

def run_pipeline(scene_name: str, config_path: str, out_dir_path: str, max_frames: int = None):
    log_stage("Pipeline Execution")
    
    out_dir = Path(out_dir_path)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    loader = NuScenesLoader(config_path)
    frames = loader.load_scene(scene_name, max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser(config_path)
    projector = IPMProjector(config_path)
    tracker = Tracker(config_path)
    builder = TopologyBuilder(config_path)
    
    logger.info("Processing frames...")
    
    for frame in frames:
        res = segmenter.segment(frame)
        polylines, _ = vectoriser.process(res.drivable_mask, res.lane_mask)
        projected = projector.project(
            polylines, 
            frame.camera_matrix, 
            frame.cam_to_ego_R, 
            frame.cam_to_ego_t
        )
        global_detections = []
        for pl in projected:
            pts_ego_3d = np.column_stack((pl.points, np.zeros(len(pl.points))))
            pts_global_3d = frame.ego_to_global(pts_ego_3d)
            pl.points = pts_global_3d[:, :2]
            global_detections.append(pl)
            
        tracker.update(global_detections)
        
    tracker.duplicate_fusion()
    all_tracks = tracker.get_all_tracks()
    
    logger.info("Building topology graph...")
    graph = builder.build_graph(all_tracks)
    
    json_path = out_dir / f"topology_{scene_name}.json"
    bev_path = out_dir / f"topology_bev_{scene_name}.png"
    vid_path = out_dir / f"reprojection_{scene_name}.mp4"
    
    logger.info(f"Serializing JSON to {json_path}...")
    validate_and_save(graph, str(json_path))
    
    generate_bev_plot(graph, frames, bev_path, scene_name)
    generate_reprojection(graph, frames, vid_path)
    
    logger.info("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help="Scene name")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--list-scenes', action='store_true', help="List available scenes")
    args = parser.parse_args()

    if args.list_scenes:
        loader = NuScenesLoader(args.config)
        scenes = loader.list_scenes()
        print("Available scenes:")
        for s in scenes:
            print(f"  - {s['name']}: {s['description']}")
        return

    if not args.scene:
        print("Error: --scene is required unless --list-scenes is used.")
        sys.exit(1)
        
    run_pipeline(args.scene, args.config, args.out, args.max_frames)

if __name__ == '__main__':
    main()
