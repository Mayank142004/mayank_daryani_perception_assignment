from pathlib import Path
import cv2
import numpy as np
import torch
import warnings
from src.segment.base import LaneSegmenter, SegmentationResult
from src.data.loader import Frame
from src.segment.letterbox import letterbox
from src.utils.logging import logger

class YOLOPv2Segmenter(LaneSegmenter):
    def __init__(self, weights_path: str = 'weights/yolopv2.pt', cache_dir: str = 'outputs/cache/masks'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading YOLOPv2 from {weights_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.jit.load(weights_path, map_location='cpu')
        self.model.eval()

    def segment(self, frame: Frame) -> SegmentationResult:
        cache_path = self.cache_dir / f"{frame.token}.npz"
        if cache_path.exists():
            data = np.load(str(cache_path))
            return SegmentationResult(
                drivable_mask=data['da_mask'],
                lane_mask=data['ll_mask']
            )

        img_orig = cv2.imread(frame.image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Image not found: {frame.image_path}")
            
        orig_h, orig_w = img_orig.shape[:2]

        # Letterbox to 384x640
        lb_img, r, pad = letterbox(img_orig, new_shape=(384, 640))
        
        # Prepare tensor
        img_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            out = self.model(img_tensor)
            _, seg, ll = out
            
        # Extract masks
        da_mask_pad = torch.argmax(seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        ll_mask_pad = (ll.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

        # Remove padding
        pad_w, pad_h = pad
        h, w = da_mask_pad.shape
        top = int(pad_h)
        bottom = int(h - pad_h)
        left = int(pad_w)
        right = int(w - pad_w)
        
        da_mask_crop = da_mask_pad[top:bottom, left:right]
        ll_mask_crop = ll_mask_pad[top:bottom, left:right]

        # Resize back to original size
        da_mask = cv2.resize(da_mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Cache masks
        np.savez_compressed(str(cache_path), da_mask=da_mask, ll_mask=ll_mask)

        return SegmentationResult(
            drivable_mask=da_mask,
            lane_mask=ll_mask
        )
