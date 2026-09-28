import yaml
import numpy as np
import uuid
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from typing import List, Tuple
from src.geometry.ipm import ProjectedPolyline
from src.utils.logging import logger

def compute_arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros(len(points))
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(diffs)))
    return s

def local_heading(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    vec = points[-1] - points[0]
    return np.arctan2(vec[1], vec[0])

@dataclass
class Track:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    confidences: np.ndarray = field(default_factory=lambda: np.zeros(0))
    misses: int = 0
    hits: int = 1
    
    @property
    def length_m(self):
        if len(self.points) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))

class Tracker:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['stitch']
        self.live_tracks: List[Track] = []
        self.archived_tracks: List[Track] = []

    def _get_track_tail(self, track: Track, n: int) -> np.ndarray:
        return track.points[-n:]

    def _get_det_head(self, det: np.ndarray, n: int) -> np.ndarray:
        return det[:n]
        
    def _heading_diff(self, a: float, b: float) -> float:
        diff = (a - b) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        return np.abs(diff)

    def _dist_point_to_polyline(self, pt: np.ndarray, polyline: np.ndarray) -> float:
        dists = np.linalg.norm(polyline - pt, axis=1)
        return np.min(dists)
        
    def match(self, detections: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian matching."""
        if not self.live_tracks or not detections:
            return [], list(range(len(self.live_tracks))), list(range(len(detections)))
            
        N = len(self.live_tracks)
        M = len(detections)
        cost_matrix = np.full((N, M), 1e6)
        
        n_pts = self.cfg['heading_n_points']
        gate_dist = self.cfg['gate_dist_m']
        gate_angle = np.deg2rad(self.cfg['gate_angle_deg'])
        w_angle = self.cfg['cost_angle_weight']
        
        for i, trk in enumerate(self.live_tracks):
            tail = self._get_track_tail(trk, n_pts)
            trk_heading = local_heading(tail)
            
            for j, det in enumerate(detections):
                head = self._get_det_head(det, n_pts)
                det_heading = local_heading(head)
                
                # Heading diff
                angle_diff = self._heading_diff(trk_heading, det_heading)
                
                # Distance from det head to track tail
                dist = np.linalg.norm(head[0] - tail[-1])
                
                # Fallback: overlap distance if they span the same region
                if dist > gate_dist:
                    # check if det head is close to any point on track
                    dist = self._dist_point_to_polyline(head[0], trk.points)
                
                if dist < gate_dist and angle_diff < gate_angle:
                    cost_matrix[i, j] = dist + w_angle * angle_diff
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                matches.append((r, c))
                
        matched_trk = [m[0] for m in matches]
        matched_det = [m[1] for m in matches]
        
        unmatched_trk = [i for i in range(N) if i not in matched_trk]
        unmatched_det = [j for j in range(M) if j not in matched_det]
        
        return matches, unmatched_trk, unmatched_det

    def fuse_detection(self, track: Track, det_points: np.ndarray, det_confs: np.ndarray):
        """
        Project detection onto track's arc length, bin at fixed spacing, average.
        Follow with Savitzky-Golay smoothing.
        """
        spacing = self.cfg['bin_spacing_m']
        
        # Combine points
        combined = np.vstack((track.points, det_points))
        combined_c = np.concatenate((track.confidences, det_confs))
        
        if len(combined) < 2:
            track.points = combined
            track.confidences = combined_c
            return

        # Compute arc length of track
        s_track = compute_arc_length(track.points)
        
        # Project det_points onto track's arc length
        # For simplicity, find nearest point on track
        s_det = []
        for p in det_points:
            dists = np.linalg.norm(track.points - p, axis=1)
            idx = np.argmin(dists)
            s_det.append(s_track[idx])
            
        s_det = np.array(s_det)
        
        # Combine
        combined_s = np.concatenate((s_track, s_det))
        combined_pts = np.vstack((track.points, det_points))
        combined_c = np.concatenate((track.confidences, det_confs))
        
        # Sort by arc length
        sort_idx = np.argsort(combined_s)
        combined_s = combined_s[sort_idx]
        combined_pts = combined_pts[sort_idx]
        combined_c = combined_c[sort_idx]
        
        # Binning
        max_arc = combined_s[-1]
        bins = np.arange(0, max_arc + spacing, spacing)
        
        fused_pts = []
        fused_c = []
        
        for i in range(len(bins) - 1):
            mask = (combined_s >= bins[i]) & (combined_s < bins[i+1])
            if np.any(mask):
                fused_pts.append(np.mean(combined_pts[mask], axis=0))
                fused_c.append(np.mean(combined_c[mask]))
                
        if len(fused_pts) == 0:
            return
            
        fused_pts = np.array(fused_pts)
        fused_c = np.array(fused_c)
        
        # Smooth
        window = min(len(fused_pts), 21)
        if window % 2 == 0:
            window -= 1
        poly = self.cfg['savgol_polyorder']
        
        if window > poly and window >= 3:
            fused_pts[:, 0] = savgol_filter(fused_pts[:, 0], window_length=window, polyorder=poly)
            fused_pts[:, 1] = savgol_filter(fused_pts[:, 1], window_length=window, polyorder=poly)
            
        track.points = fused_pts
        track.confidences = fused_c

    def update(self, detections_global: List[ProjectedPolyline]):
        dets_pts = [d.points for d in detections_global]
        matches, unmatched_trk, unmatched_det = self.match(dets_pts)
        
        # Update matches
        for trk_idx, det_idx in matches:
            trk = self.live_tracks[trk_idx]
            det_pts = detections_global[det_idx].points
            det_confs = detections_global[det_idx].confidences
            
            self.fuse_detection(trk, det_pts, det_confs)
            trk.hits += 1
            trk.misses = 0
            
        # Unmatched tracks get a miss
        new_archived = 0
        alive = []
        for trk_idx in unmatched_trk:
            trk = self.live_tracks[trk_idx]
            trk.misses += 1
            if trk.misses > self.cfg['max_misses']:
                self.archived_tracks.append(trk)
                new_archived += 1
            else:
                alive.append(trk)
                
        # Matched tracks stay alive
        for m in matches:
            alive.append(self.live_tracks[m[0]])
            
        self.live_tracks = alive
        
        # Spawn new tracks
        for det_idx in unmatched_det:
            det = detections_global[det_idx]
            trk = Track(points=det.points.copy(), confidences=det.confidences.copy())
            self.live_tracks.append(trk)
            
        return len(self.live_tracks), len(matches), len(unmatched_det), new_archived

    def get_all_tracks(self) -> List[Track]:
        return self.archived_tracks + self.live_tracks

    def duplicate_fusion(self):
        """Pass after all frames to merge duplicates."""
        all_tracks = self.get_all_tracks()
        if not all_tracks:
            return
            
        gate_dist = self.cfg['duplicate_gate_dist_m']
        gate_angle = np.deg2rad(self.cfg['duplicate_gate_angle_deg'])
        
        merged = []
        skip = set()
        
        for i, trk1 in enumerate(all_tracks):
            if i in skip:
                continue
            
            base = Track(id=trk1.id, points=trk1.points.copy(), confidences=trk1.confidences.copy(), hits=trk1.hits)
            
            for j in range(i + 1, len(all_tracks)):
                if j in skip:
                    continue
                trk2 = all_tracks[j]
                
                # Check overlap and heading
                if len(base.points) < 2 or len(trk2.points) < 2:
                    continue
                    
                h1 = local_heading(base.points)
                h2 = local_heading(trk2.points)
                if self._heading_diff(h1, h2) > gate_angle:
                    continue
                    
                # Mean distance over overlapping region
                # For simplicity here: check if mean distance from trk2 points to base is small
                dists = [self._dist_point_to_polyline(p, base.points) for p in trk2.points]
                if np.mean(dists) < gate_dist:
                    self.fuse_detection(base, trk2.points, trk2.confidences)
                    base.hits += trk2.hits
                    skip.add(j)
                    
            merged.append(base)
            
        self.archived_tracks = merged
        self.live_tracks = []
