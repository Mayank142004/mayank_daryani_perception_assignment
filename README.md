# nuScenes Lane Topology Extraction

This pipeline takes nuScenes v1.0-mini `CAM_FRONT` keyframes and extracts a 3D road network graph in a global coordinate frame. It segments images using a CPU-bound YOLOPv2 model, vectorises the lane and drivable-area masks, and tracks the markings over time. The output is a serialized, JSON-validated topology graph comprising drivable centrelines (nodes) and their semantic connections (edges).

## Results

The pipeline was run end-to-end on `scene-0061` (39 frames). 

| Metric | Measured Value |
|--------|----------------|
| **Graph Nodes** | 12 |
| **Graph Edges** | 7 |
| **Edge Breakdown** | 4 continuation, 3 junction |
| **Total Road Length** | 68.29 m |
| **Mean Turn per Point** | 3.55 degrees |
| **Points with Kink > 30°** | 1.78% |
| **Confidence (min / mean / max)** | 0.050 / 0.336 / 0.542 |
| **Curvature (min / max)** | 0.0000 / 0.6447 |

Only `continuation` and `junction` edges were produced in this run. No `split` or `merge` edges were detected. The boundary-pairing logic rigidly requires two parallel boundaries to construct a centreline, which leaves few surviving tracks dense enough to fork or converge. This is a limitation of the current pairing tolerance.

The Chamfer-distance evaluation against the nuScenes HD map was not computed. The `v1.0-mini` nuScenes dataset does not include the required vector map expansion pack (`singapore-onenorth.json`), making quantitative evaluation against the ground truth impossible without an external download.

## Install

Requires Python 3.10 and a CPU-only environment.

```bash
# Install PyTorch for CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

Unpack the `nuScenes v1.0-mini` dataset into `data/nuscenes`. Download the pretrained YOLOPv2 weights (BDD100K) from the official YOLOPv2 repository and place the file at `weights/yolopv2.pt`.

## Run

Run the single end-to-end pipeline entry point. This automatically executes all stages, serializes the JSON, generates a BEV map, and renders the reprojection video:

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py --scene scene-0061
```

To list available scenes:
```bash
PYTHONPATH=. python3 scripts/run_pipeline.py --list-scenes
```

For targeted debugging of intermediate stages, use the following scripts:

| Script | Output Produced |
|--------|-----------------|
| `scripts/inspect_scene.py` | Extracts and prints basic camera tracking and ego pose diagnostics. |
| `scripts/debug_masks.py` | Overlays YOLOPv2 drivable and lane masks onto the original images. |
| `scripts/debug_vectorise.py` | Visualises the 2D ordered polylines after skeletonisation and drivable-area gating. |
| `scripts/debug_project.py` | Renders a BEV plot of the independent frame detections inverse-projected to the ground plane. |
| `scripts/debug_stitch.py` | Renders a global BEV plot of the temporally stitched boundary tracks. |
| `scripts/debug_topology.py` | Wrapper for `run_pipeline.py` that validates and serializes the JSON graph. |

## Architecture

```text
[Ingest] -> [Segment] -> [Vectorise] -> [Project] -> [Stitch] -> [Topology] -> [Serialize]
```

- **Ingest:** Parses the nuScenes dataset, resolving keyframe images, intrinsics, and ego poses.
- **Segment:** Letterboxes the image and runs a PyTorch CPU YOLOPv2 forward pass, extracting lane and drivable-area masks.
- **Vectorise:** Gates the lanes using the dilated drivable area, then applies morphological skeletonisation and pixel-ordering to extract 2D polylines.
- **Project:** Applies Inverse Perspective Mapping (IPM) ray intersection to cast the 2D pixels onto the ego vehicle's $Z=0$ flat ground plane.
- **Stitch:** Uses a Hungarian algorithm to track boundaries globally, projecting and smoothing new detections onto a cumulative arc length.
- **Topology:** Pairs parallel boundaries into centrelines, breaks them into nodes, and infers semantic edges (`continuation`, `split`, `merge`, `junction`).
- **Serialize:** Validates the final graph against a rigorous jsonschema before writing to disk.

Read the expanded design document in `docs/architecture.md`.

## Why segmentation

This pipeline uses a segmentation-based approach rather than a row-anchor architecture (like UFLDv2). YOLOPv2 is a multi-task network with parallel heads. This allows us to extract both the lane mask and the drivable-area mask from a single forward pass. We use the drivable-area mask to actively gate the lane candidates, successfully rejecting false positives occurring on kerbs and barriers.

Segmentation was chosen over a row-anchor model because row-anchor models inherently impose a structural cap on the maximum number of detectable lanes. Furthermore, row anchors are typically calibrated to the camera crop of a specific dataset, making them rigid when transferring across domains. The pixel-level output of a segmentation model allows theoretically unbounded lane discovery.

## Coordinate frames and output schema

- **Camera frame:** X right, Y down, Z forward.
- **Ego frame:** X forward, Y left, Z up. Origin at the vehicle's rear axle.
- **Global frame:** Standard metric global coordinate space initialized at the scene's first pose.

All distance and coordinate metrics are in meters.

```json
{
  "nodes": [
    {
      "id": "node_7e3a9c1b",
      "polyline": [
        [371.42, 1123.85],
        [371.10, 1124.23]
      ],
      "attributes": {
        "direction": "115.3 deg",
        "curvature": 0.0412,
        "confidence": 0.381,
        "n_observations": 12,
        "length_m": 4.5,
        "mean_range_m": 12.3
      }
    }
  ],
  "edges": [
    {
      "from": "node_7e3a9c1b",
      "to": "node_9b4e1d2a",
      "type": "continuation",
      "confidence": 1.0
    }
  ]
}
```

- `direction`: General heading of the segment tail, in degrees.
- `curvature`: Mean curvature computed over a sliding window using the Menger formula.
- `confidence`: Bounded multiplier factoring segment length completeness, observation counts, and range-decayed pixel probability.
- `n_observations`: Total stitched point count across frames making up the segment.
- `length_m`: Physical arc length of the centreline segment in meters.
- `mean_range_m`: The mean distance of the segment's observations from the ego vehicle.

## Design decisions

- **Keyframes at 2 Hz:** We only ingest keyframes rather than the full 12 Hz video sweeps. This massively reduces CPU inference load while still guaranteeing a detection every ~5 meters, but slightly sacrifices temporal smoothness during tracking.
- **Centrelines vs Boundaries:** The pipeline averages paired lane boundaries to emit drivable centrelines. This aligns the output with planner routing needs, but aggressively discards any single boundary that fails to find a valid parallel pair.
- **Drivable-area gating:** We dilate the drivable area and use it to mask the lane probabilities. This eliminates kerb and barrier noise, but risks truncating valid outer lane lines if the drivable-area prediction is overly conservative.
- **Flat-ground IPM:** We inverse-project pixels by intersecting rays against the ego vehicle's $Z=0$ plane. This avoids the need for external depth estimators, but guarantees the projection will break heavily on sloped roads, banked turns, or bumps.

## Known limitations

- Flat-ground IPM geometry inherently diverges on gradients and banked turns, with the error growing exponentially with range.
- The YOLOPv2 model is pretrained on BDD100K. The shift to the nuScenes domain (different camera intrinsics and mounting height) results in mask degradation.
- Boundary pairing yields low graph coverage compared to what a full scene actually contains.
- No `split` or `merge` edges were produced in the current outputs due to the strict pairing heuristics.
- No quantitative accuracy metric is currently computed against the HD map.

See `docs/assumptions.md` for a full list of addressed failure modes.

## Testing

The test suite enforces the core mathematical and geometric constraints of the pipeline.

- `tests/test_ipm.py`: Asserts inverse projection and re-projection roundtrips perfectly under 1 pixel error.
- `tests/test_letterbox.py`: Asserts image letterboxing and point un-letterboxing maps backwards precisely under 2 pixels error.
- `tests/test_ordering.py`: Asserts skeleton pixel blobs are grouped and strictly sorted near-to-far.
- `tests/test_stitch.py`: Asserts arc-length fusion and smoothing prevents kink sawteeth, keeping the mean turn-per-point under 5 degrees.
- `tests/test_topology.py`: Asserts geometric heuristics correctly classify all four semantic edge types (`continuation`, `split`, `merge`, `junction`).

To run the suite:
```bash
PYTHONPATH=. pytest -v
```

Test Results:
```text
============================== 5 passed in 3.24s ===============================
```

## Repository layout

```text
src/
├── data/loader.py           # nuScenes dataset parsing and coordinate transforms
├── segment/base.py          # Abstract segmentation interfaces
├── segment/letterbox.py     # Deterministic image resizing and point mapping
├── segment/yolopv2.py       # YOLOPv2 inference execution and caching
├── vectorise/mask_to_lines.py # Skeletonisation, component filtering, and ordering
├── geometry/ipm.py          # Inverse Perspective Mapping (flat-ground)
├── stitch/temporal.py       # Multi-object tracking and arc-length fusion
├── graph/topology.py        # Centreline pairing, node splitting, and edge inference
└── graph/schema.py          # jsonschema definition and validation

scripts/
├── inspect_scene.py         # Print basic scene metadata and poses
├── phase0_feasibility.py    # Standalone script for YOLOPv2 viability check
├── debug_masks.py           # Render raw mask video
├── debug_vectorise.py       # Render 2D pixel polyline video
├── debug_project.py         # Render disconnected 3D IPM scatter plots
├── debug_stitch.py          # Render global stitched tracks BEV map
├── debug_topology.py        # Wrapper to serialize the topology graph
├── metrics.py               # Headless pipeline execution and metric logging
├── visualise.py             # Reprojection video rendering
├── evaluate.py              # HD Map Chamfer distance evaluator (requires map pack)
└── run_pipeline.py          # Canonical end-to-end runner (JSON, BEV, Reprojection)
```
