import yaml
import numpy as np
from typing import List, Dict, Tuple, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import uuid
import networkx as nx
from dataclasses import dataclass

from src.stitch.temporal import Track, local_heading, compute_arc_length
from src.graph.schema import GraphNode, GraphEdge, NodeAttributes, TopologyGraph
from src.utils.logging import logger

def menger_curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p1 - p3)
    if a == 0 or b == 0 or c == 0:
        return 0.0
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return 0.0
    area = np.sqrt(area_sq)
    return 4 * area / (a * b * c)

def compute_track_curvature(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    # Sliding window of 3 points, step by len//3 to get overall curvature
    curvatures = []
    step = max(1, len(pts) // 5)
    for i in range(0, len(pts) - 2 * step, step):
        p1, p2, p3 = pts[i], pts[i + step], pts[i + 2 * step]
        curvatures.append(menger_curvature(p1, p2, p3))
    return float(np.mean(curvatures)) if curvatures else 0.0

@dataclass
class Centreline:
    points: np.ndarray
    confidences: np.ndarray

class TopologyBuilder:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['topology']

    def pair_boundaries(self, tracks: List[Track]) -> List[Centreline]:
        if len(tracks) < 2:
            return []
            
        N = len(tracks)
        cost_matrix = np.full((N, N), 1e6)
        
        min_w = self.cfg['min_lane_width']
        max_w = self.cfg['max_lane_width']
        min_overlap = self.cfg['min_overlap_m']
        
        for i in range(N):
            pts_i = tracks[i].points
            if len(pts_i) < 2:
                continue
            for j in range(i + 1, N):
                pts_j = tracks[j].points
                if len(pts_j) < 2:
                    continue
                    
                dists = cdist(pts_i, pts_j) # (N_i, N_j)
                min_dists_i = np.min(dists, axis=1) # (N_i,)
                
                # Overlapping points in i
                overlap_mask = (min_dists_i >= min_w) & (min_dists_i <= max_w)
                if not np.any(overlap_mask):
                    continue
                    
                overlap_pts_i = pts_i[overlap_mask]
                overlap_len = compute_arc_length(overlap_pts_i)[-1] if len(overlap_pts_i) > 1 else 0.0
                
                if overlap_len >= min_overlap:
                    cost_matrix[i, j] = np.mean(min_dists_i[overlap_mask])
                    cost_matrix[j, i] = cost_matrix[i, j]
                    
        # Match
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        centrelines = []
        paired = set()
        
        for r, c in zip(row_ind, col_ind):
            if r in paired or c in paired:
                continue
            if cost_matrix[r, c] < 1e5:
                # Build centreline
                pts_r = tracks[r].points
                pts_c = tracks[c].points
                
                # Resample r to compute c
                dists = cdist(pts_r, pts_c)
                min_idx_c = np.argmin(dists, axis=1)
                min_dists_r = np.min(dists, axis=1)
                
                mask = (min_dists_r >= min_w) & (min_dists_r <= max_w)
                
                if np.sum(mask) > 1:
                    valid_r = pts_r[mask]
                    valid_c = pts_c[min_idx_c[mask]]
                    
                    c_points = (valid_r + valid_c) / 2.0
                    c_confs = (tracks[r].confidences[mask] + tracks[c].confidences[min_idx_c[mask]]) / 2.0
                    
                    centrelines.append(Centreline(points=c_points, confidences=c_confs))
                    
                paired.add(r)
                paired.add(c)
                
        return centrelines

    def split_into_nodes(self, centrelines: List[Centreline]) -> List[GraphNode]:
        nodes = []
        seg_len = self.cfg['segment_length_m']
        
        for cl in centrelines:
            s = compute_arc_length(cl.points)
            if s[-1] < 1.0:
                continue
                
            num_segs = max(1, int(np.ceil(s[-1] / seg_len)))
            bins = np.linspace(0, s[-1], num_segs + 1)
            
            for i in range(num_segs):
                mask = (s >= bins[i]) & (s <= bins[i+1])
                if np.sum(mask) >= 2:
                    pts = cl.points[mask]
                    confs = cl.confidences[mask]
                    
                    length = compute_arc_length(pts)[-1]
                    curv = compute_track_curvature(pts)
                    n_obs = len(pts) # Rough proxy
                    
                    # F10: confidence computed as product of observation, completeness, range
                    obs_factor = min(1.0, n_obs / 10.0)
                    comp_factor = min(1.0, length / seg_len)
                    range_factor = np.mean(confs) # Assuming input confs already decayed by range
                    
                    confidence = float(np.clip(obs_factor * comp_factor * range_factor, 0.05, 1.0))
                    
                    # Direction: string representing general heading (N, S, E, W) roughly
                    # Actually local_heading is radians [-pi, pi]. Just save radians as string or degrees?
                    # "direction: string"
                    h = local_heading(pts)
                    direction = f"{np.rad2deg(h):.1f} deg"
                    
                    attr = NodeAttributes(
                        direction=direction,
                        curvature=curv,
                        confidence=confidence,
                        n_observations=n_obs,
                        length_m=float(length),
                        mean_range_m=float(np.mean(np.linalg.norm(pts, axis=1)))
                    )
                    
                    nodes.append(GraphNode(
                        id=f"node_{str(uuid.uuid4())[:8]}",
                        polyline=pts.tolist(),
                        attributes=attr
                    ))
        return nodes

    def build_edges(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        edges = []
        
        gate_dist = self.cfg['continuation_gate_dist']
        gate_angle = np.deg2rad(self.cfg['continuation_gate_angle_deg'])
        sm_rad = self.cfg['split_merge_radius']
        sm_angle = np.deg2rad(self.cfg['split_merge_angle_deg'])
        j_rad = self.cfg['junction_search_radius']
        j_angle = np.deg2rad(self.cfg['junction_angle_deg'])
        
        def _head(n): return np.array(n.polyline[0])
        def _tail(n): return np.array(n.polyline[-1])
        def _heading(n): return local_heading(np.array(n.polyline))
        def _angle_diff(h1, h2):
            d = (h1 - h2) % (2 * np.pi)
            if d > np.pi: d -= 2 * np.pi
            return abs(d)

        has_continuation = set()
        
        for i, n_i in enumerate(nodes):
            tail_i = _tail(n_i)
            h_i = _heading(n_i)
            
            dists = []
            for j, n_j in enumerate(nodes):
                if i == j: continue
                head_j = _head(n_j)
                h_j = _heading(n_j)
                d = np.linalg.norm(tail_i - head_j)
                a = _angle_diff(h_i, h_j)
                dists.append((j, d, a))
                
            if not dists: continue
            
            # Sort by distance
            dists.sort(key=lambda x: x[1])
            
            # 1. Split
            sm_cands = [x for x in dists if x[1] < sm_rad and x[2] < sm_angle]
            if len(sm_cands) >= 2:
                for cand in sm_cands[:2]:
                    edges.append(GraphEdge(from_node=n_i.id, to_node=nodes[cand[0]].id, type="split", confidence=0.8))
                has_continuation.add(n_i.id)
                continue
                
            # 2. Continuation
            continuation_cands = [x for x in dists if x[1] < gate_dist and x[2] < gate_angle]
            if continuation_cands:
                j = continuation_cands[0][0]
                edges.append(GraphEdge(from_node=n_i.id, to_node=nodes[j].id, type="continuation", confidence=1.0))
                has_continuation.add(n_i.id)
                continue
                
            # Merge is implicit if multiple nodes point to the same next node. 
            # We will handle merge by looking backwards.
            
            # 3. Junction
            # Where centreline ends with no continuation, search wider radius
            j_cands = [x for x in dists if x[1] < j_rad and x[2] < j_angle]
            if j_cands:
                j = j_cands[0][0]
                edges.append(GraphEdge(from_node=n_i.id, to_node=nodes[j].id, type="junction", confidence=0.5))
                has_continuation.add(n_i.id)
                
        # Fix merges: if two edges point to the same node with 'continuation', change them to 'merge'
        to_counts = {}
        for e in edges:
            if e.type in ('continuation', 'split'):
                to_counts[e.to_node] = to_counts.get(e.to_node, 0) + 1
                
        for e in edges:
            if e.type == 'continuation' and to_counts.get(e.to_node, 0) > 1:
                e.type = 'merge'
                
        return edges

    def build_graph(self, tracks: List[Track]) -> TopologyGraph:
        centrelines = self.pair_boundaries(tracks)
        nodes = self.split_into_nodes(centrelines)
        edges = self.build_edges(nodes)
        return TopologyGraph(nodes=nodes, edges=edges)

    def validate_graph(self, G: TopologyGraph):
        nx_g = nx.DiGraph()
        for n in G.nodes:
            nx_g.add_node(n.id)
        for e in G.edges:
            nx_g.add_edge(e.from_node, e.to_node, type=e.type)
            
        orphans = [n for n in nx_g.nodes if nx_g.degree(n) == 0]
        cycles = list(nx.simple_cycles(nx_g))
        components = nx.number_weakly_connected_components(nx_g)
        
        logger.info(f"Graph Validation:")
        logger.info(f"- Nodes: {len(G.nodes)}")
        logger.info(f"- Edges: {len(G.edges)}")
        logger.info(f"- Orphan nodes: {len(orphans)}")
        logger.info(f"- Cycles found: {len(cycles)}")
        logger.info(f"- Connected components: {components}")
        
        return nx_g
