from typing import Protocol
from dataclasses import dataclass
import numpy as np
from src.data.loader import Frame

@dataclass
class SegmentationResult:
    drivable_mask: np.ndarray  # (H, W) uint8
    lane_mask: np.ndarray      # (H, W) uint8

class LaneSegmenter(Protocol):
    def segment(self, frame: Frame) -> SegmentationResult:
        """Process a frame and return drivable and lane masks."""
        ...
