import cv2
import numpy as np
from typing import Tuple

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (384, 640), color: Tuple[int, int, int] = (114, 114, 114)):
    """
    Resize image to a multiple of stride (new_shape) maintaining aspect ratio.
    new_shape is (H, W).
    Returns:
        img: padded image
        r: resize ratio
        pad: (pad_w, pad_h) applied to left/right and top/bottom
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (left, top)

def unletterbox_points(points: np.ndarray, r: float, pad: Tuple[float, float]) -> np.ndarray:
    """
    Map points from letterboxed image back to original image coordinates.
    points: (N, 2) array of (x, y)
    r: resize ratio returned by letterbox
    pad: (pad_w, pad_h) returned by letterbox
    """
    pad_w, pad_h = pad
    unpadded = points - np.array([pad_w, pad_h])
    return unpadded / r
