# Pipeline Architecture

This document outlines the data flow and architectural components of the segmentation-based lane topology extraction pipeline.

## 1. Ingest (`src/data/loader.py`)
The pipeline consumes nuScenes `CAM_FRONT` keyframes. We operate strictly at 2 Hz (keyframes only) to aggressively cut inference overhead while guaranteeing a detection approximately every ~5 meters at urban driving speeds. Extrinsics (translations and quaternions) are converted into rotation matrices once at load time, producing a clean `Frame` dataclass that exposes direct `ego_to_global()` and `global_to_ego()` coordinate transformations.

## 2. Segment (`src/segment/yolopv2.py`)
Raw 1600x900 imagery undergoes rigid letterboxing to a 640x384 tensor, feeding a PyTorch CPU-bound YOLOPv2 instance. Operating entirely in `eval()` mode with `torch.no_grad()`, a single forward pass yields parallel drivable-area and lane-line segmentation heads. We cache the argmaxed drivable mask and the raw float32 lane probabilities to disk per-frame, preventing redundant computation during downstream tuning.

## 3. Vectorise (`src/vectorise/mask_to_lines.py`)
This stage bridges the pixel and vector domains.
1. The drivable area mask is dilated to prevent overly aggressive boundary clipping.
2. The lane probabilities are thresholded, gated by the drivable area, and passed through `skimage.morphology.skeletonize`.
3. Connected components are labeled and filtered via strict morphological rules (pixel counts, vertical extents, and width/height ratios) to eliminate stop lines and kerb noise.
4. Surviving pixel sets are grouped by row, collapsing into median columns, and emitted strictly near-to-far (bottom-of-image to top).

## 4. Project (`src/geometry/ipm.py`)
Pixels are inverse-projected onto a flat ground plane using the loaded camera intrinsics ($K$) and extrinsics ($R, t$). Rays are cast from the optical center and intersected against the $Z=0$ ego-frame plane. Points falling behind the camera or extending beyond a configured horizon limit are safely discarded. A confidence decay is applied based on the Euclidean radial distance from the vehicle.

## 5. Stitch (`src/stitch/temporal.py`)
Ego-frame polylines are mapped into global coordinates. A multi-object tracker matches incoming detections to live tracks using a bipartite Hungarian algorithm. The cost matrix dynamically evaluates distance and angular disparity. Fused detections are projected directly onto the host track's cumulative arc length, sorted, uniformly binned at 0.5m spacing, and smoothed via a Savitzky-Golay filter to ensure structurally smooth trajectories.

## 6. Topology (`src/graph/topology.py`)
The tracker outputs raw lane boundaries. These boundaries are compared using an overlapping-region nearest-neighbor calculation. Parallel boundaries falling within a legitimate lane-width band are mathematically averaged into unified drivable centrelines. 
These centrelines are chopped into discrete nodes, and geometrical heuristics bind them via semantically typed edges (`continuation`, `split`, `merge`, `junction`). Node attributes (including Menger curvature) are actively calculated before final schema validation.
