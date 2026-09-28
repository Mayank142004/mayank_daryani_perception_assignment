from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
scene = nusc.scene[0]
print("Scene name:", scene['name'])

current_token = scene['first_sample_token']
poses = []
c_heights = []
fxs = []
frames = 0
while current_token:
    sample = nusc.get('sample', current_token)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    K = np.array(calib['camera_intrinsic'])
    c_heights.append(calib['translation'][2])
    fxs.append(K[0,0])
    
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    poses.append(np.array(ego_pose['translation']))
    
    frames += 1
    current_token = sample['next']

dist = sum(np.linalg.norm(poses[i] - poses[i-1]) for i in range(1, len(poses)))
print("Frames:", frames)
print("Mean camera height:", np.mean(c_heights))
print("Mean fx:", np.mean(fxs))
print("Total distance:", dist)
