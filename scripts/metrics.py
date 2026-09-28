import argparse
import time
import numpy as np
from collections import Counter
import warnings

from src.data.loader import NuScenesLoader
from src.segment.yolopv2 import YOLOPv2Segmenter
from src.vectorise.mask_to_lines import Vectoriser
from src.geometry.ipm import IPMProjector
from src.stitch.temporal import Tracker
from src.graph.topology import TopologyBuilder
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()
    
    log_stage("Metrics Pipeline")
    
    loader = NuScenesLoader('config.yaml')
    frames = loader.load_scene(args.scene)
    if not frames:
        return
        
    segmenter = YOLOPv2Segmenter()
    vectoriser = Vectoriser('config.yaml')
    projector = IPMProjector('config.yaml')
    tracker = Tracker('config.yaml')
    builder = TopologyBuilder('config.yaml')
    
    total_rejects = {'mask_gate': 0, 'min_pixels': 0, 'min_vert_extent': 0, 'max_wh_ratio': 0}
    
    t0 = time.perf_counter()
    
    for frame in frames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = segmenter.segment(frame)
            polylines, rejects = vectoriser.process(res.drivable_mask, res.lane_mask)
            
            for k, v in rejects.items():
                total_rejects[k] += v
                
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
    graph = builder.build_graph(all_tracks)
    
    t1 = time.perf_counter()
    fps = len(frames) / (t1 - t0)
    
    nodes = graph.nodes
    edges = graph.edges
    
    length_m = sum(n.attributes.length_m for n in nodes if n.attributes.length_m)
    curvatures = [n.attributes.curvature for n in nodes if n.attributes.curvature is not None]
    confs = [n.attributes.confidence for n in nodes if n.attributes.confidence is not None]
    
    mean_curv = np.mean(curvatures) if curvatures else 0.0
    mean_conf = np.mean(confs) if confs else 0.0
    
    edge_types = Counter(e.type for e in edges)
    
    print("\n" + "="*40)
    print("PIPELINE METRICS")
    print("="*40)
    print(f"Graph Nodes:           {len(nodes)}")
    print(f"Graph Edges:           {len(edges)}")
    for t, c in edge_types.items():
        print(f"  - {t}: {c}")
    print(f"Total length:          {length_m:.2f} m")
    print(f"Mean node curvature:   {mean_curv:.4f}")
    print(f"Mean confidence:       {mean_conf:.3f}")
    print(f"Vectorise Rejections:  {total_rejects}")
    print(f"Pipeline Runtime:      {fps:.2f} FPS")
    print("="*40)

if __name__ == '__main__':
    main()
