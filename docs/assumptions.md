# Design Assumptions and Failure Mode Mitigation

This document clarifies critical design decisions and outlines how the pipeline structurally mitigates the eleven known historical failure modes associated with lane topology extraction.

## Centrelines vs. Boundaries

**Design Decision:** The YOLOPv2 segmentation heads explicitly predict painted lane boundaries. However, the final JSON schema expects standard road segments, and autonomous planners fundamentally require drivable trajectories (centrelines), not paint stripes. 

To resolve this, the topology builder executes a boundary-pairing pass. It identifies roughly parallel boundaries that exist within a configurable width band (`min_lane_width` to `max_lane_width`) and averages their overlapping coordinates to extract the true drivable centreline. Unpaired boundary tracks are actively dropped from the final topology graph.

## Known Failure Modes Mitigated

1. **F1 — Discarding dead tracks:** Tracks that miss association limits are moved to an `archived_tracks` list. The topology graph is constructed using both live and archived tracks, guaranteeing no historical data is silently dropped.
2. **F2 — Structural cap on lane count:** Lane extraction uses morphological skeletonisation and connected-component labeling. There is no structural cap, top-k selection, or left/right bifurcation logic anywhere in the pipeline.
3. **F3 — Heading compared over the whole track:** The `local_heading` utility calculates headings strictly using the final $N$ points of a track's tail and the first $N$ points of a detection's head.
4. **F4 — Metres and radians in one threshold:** Distance and angular deviation are evaluated independently against distinct hard gates. A detection must satisfy both gates before a combined Euclidean-angular cost is applied to the Hungarian matrix.
5. **F5 — Dead config knobs:** All physical thresholds, filter dimensions, and logic gates are injected dynamically from `config.yaml` during class instantiation. No literal threshold values exist in the application code.
6. **F6 — Clamping points behind the camera:** The Inverse Perspective Mapping (IPM) explicitly evaluates the sign of the camera-ray's depth intersection. Rays parallel to or pointing away from the ground plane are explicitly discarded.
7. **F7 — Merging polylines by distance from first point:** Temporal tracking projects incoming detection points directly onto the cumulative arc length of the host track. Points are subsequently sorted by arc length, uniformly binned, and smoothed, completely preventing the sawtooth interleaving effect.
8. **F8 — Pairing boundaries on endpoints only:** The topology boundary pairing relies on `scipy.spatial.distance.cdist` to measure the nearest-neighbor Euclidean distance across the entire overlapping region of two tracks, ignoring unmatched endpoints.
9. **F9 — Duplicate tracks after occlusion:** A `duplicate_fusion()` pass executes globally after all frames are processed, merging any overlapping track fragments sharing identical headings.
10. **F10 — Placeholder attributes:** Node attributes are mathematically derived. Curvature leverages a sliding-window Menger formula. Confidence is calculated as a product of segment completeness, observation count, and the range-decayed pixel probabilities.
11. **F11 — Inverted or unused config flags:** Pipeline configurations exclusively utilize booleans for toggles and specific enums for logic paths (e.g., `continuation`, `split`, `merge`, `junction`). No exceptions are used for control flow.
