import argparse
from pathlib import Path
import json

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.geometry.ipm import IPMProjector
from src.stitch.temporal import Tracker
from src.graph.topology import TopologyBuilder
from src.graph.schema import validate_and_save
from src.utils.logging import logger, log_stage
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    log_stage("Topology & Serialize")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.error("No frames to process.")
        return

    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser('config.yaml')
    projector = IPMProjector('config.yaml')
    tracker = Tracker('config.yaml')
    builder = TopologyBuilder('config.yaml')
    
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
    
    logger.info("Validating graph structure...")
    builder.validate_graph(graph)
    
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"topology_{args.scene}.json"
    
    logger.info(f"Serializing and validating JSON against schema to {out_path}...")
    validate_and_save(graph, str(out_path))
    
    logger.info("Done.")

if __name__ == '__main__':
    main()
