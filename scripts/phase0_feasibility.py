import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from skimage.morphology import skeletonize
from skimage.measure import label

def run_phase0():
    # Setup directories
    out_dir = Path('outputs/phase0')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print("Loading YOLOPv2...")
    model = torch.jit.load('weights/yolopv2.pt', map_location='cpu')
    model.eval()

    # 2. Initialize NuScenes
    print("Initializing NuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
    
    # 3. Extract 10 CAM_FRONT keyframes
    # We can just iterate samples until we have 10
    sample_tokens = []
    scene = nusc.scene[0]
    current_token = scene['first_sample_token']
    while current_token and len(sample_tokens) < 10:
        sample_tokens.append(current_token)
        sample_rec = nusc.get('sample', current_token)
        current_token = sample_rec['next']

    print(f"Extracted {len(sample_tokens)} keyframes.")

    # Process each frame
    for i, token in enumerate(sample_tokens):
        sample = nusc.get('sample', token)
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img_path = nusc.get_sample_data_path(cam_front_data['token'])
        
        # Load image
        img_orig = cv2.imread(str(img_path))
        orig_h, orig_w = img_orig.shape[:2] # 900, 1600

        # Resize to 640x360
        img_resized = cv2.resize(img_orig, (640, 360))
        
        # Letterbox padding to 640x384
        img_pad = cv2.copyMakeBorder(img_resized, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Prepare for model
        # BGR to RGB, HWC to CHW
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # 1, 3, 384, 640
        
        # Inference
        t0 = time.time()
        with torch.no_grad():
            out = model(img_tensor)
            [pred, anchor_grid], seg, ll = out
        inf_time = time.time() - t0

        # Process masks
        # seg: [1, 2, 384, 640]
        # ll: [1, 1, 384, 640]
        da_mask_pad = torch.argmax(seg, dim=1).squeeze().numpy().astype(np.uint8)
        ll_mask_pad = (ll.squeeze() > 0.5).numpy().astype(np.uint8)

        # Remove padding (12 top, 12 bottom)
        da_mask_crop = da_mask_pad[12:372, :]
        ll_mask_crop = ll_mask_pad[12:372, :]

        # Resize back to original
        da_mask = cv2.resize(da_mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Compute metrics
        da_coverage = (da_mask.sum() / (orig_w * orig_h)) * 100
        ll_coverage = (ll_mask.sum() / (orig_w * orig_h)) * 100
        
        # Skeletonise and components
        ll_skeleton = skeletonize(ll_mask > 0)
        labeled_ll, num_components = label(ll_skeleton, return_num=True)

        print(f"Frame {i+1}/10: Inference {inf_time:.3f}s | "
              f"Drivable coverage: {da_coverage:.1f}% | "
              f"Lane components: {num_components}")

        # Overlays
        # Original image
        cv2.imwrite(str(out_dir / f"{i:02d}_0_original.jpg"), img_orig)
        
        # Drivable area overlay (green)
        da_overlay = img_orig.copy()
        da_overlay[da_mask == 1] = da_overlay[da_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(str(out_dir / f"{i:02d}_1_drivable.jpg"), da_overlay)

        # Lane mask overlay (red)
        ll_overlay = img_orig.copy()
        ll_overlay[ll_mask == 1] = ll_overlay[ll_mask == 1] * 0.5 + np.array([0, 0, 255]) * 0.5
        cv2.imwrite(str(out_dir / f"{i:02d}_2_lane.jpg"), ll_overlay)

if __name__ == '__main__':
    run_phase0()
