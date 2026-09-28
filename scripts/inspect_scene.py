import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import NuScenesLoader
from src.utils.logging import logger, log_stage

def main():
    parser = argparse.ArgumentParser(description="Inspect a NuScenes scene.")
    parser.add_argument('--list-scenes', action='store_true', help="List all available scenes")
    parser.add_argument('--scene', type=str, help="Name of the scene to load")
    parser.add_argument('--max-frames', type=int, default=None, help="Max frames to load")
    args = parser.parse_args()

    log_stage("Ingest")
    loader = NuScenesLoader('config.yaml')

    if args.list_scenes:
        scenes = loader.list_scenes()
        print("\nAvailable scenes:")
        for s in scenes:
            print(f"- {s['name']}: {s['description']}")
        return

    if not args.scene:
        logger.error("Must specify --scene or --list-scenes")
        return

    frames = loader.load_scene(args.scene, args.max_frames)
    
    if not frames:
        logger.warning("No frames loaded.")
        return

    c_heights = [f.cam_to_ego_t[2] for f in frames]
    fxs = [f.camera_matrix[0, 0] for f in frames]
    poses = [f.ego_to_global_t for f in frames]

    dist = sum(np.linalg.norm(poses[i] - poses[i-1]) for i in range(1, len(poses)))

    print("\n" + "="*30)
    print(f"Scene: {args.scene}")
    print(f"Keyframes: {len(frames)}")
    print(f"Mean camera height: {np.mean(c_heights):.3f} m")
    print(f"Mean fx: {np.mean(fxs):.3f} px")
    print(f"Total distance travelled: {dist:.3f} m")
    print("="*30 + "\n")

    # Plot ego trajectory
    poses_np = np.array(poses)
    plt.figure(figsize=(8, 8))
    plt.plot(poses_np[:, 0], poses_np[:, 1], marker='o', markersize=4, linestyle='-', label='Ego Path')
    plt.plot(poses_np[0, 0], poses_np[0, 1], marker='*', color='green', markersize=10, label='Start')
    plt.plot(poses_np[-1, 0], poses_np[-1, 1], marker='X', color='red', markersize=10, label='End')
    plt.title(f"Ego Trajectory - {args.scene}")
    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"trajectory_{args.scene}.png"
    plt.savefig(out_path)
    plt.close()
    
    logger.info(f"Saved ego trajectory plot to {out_path}")

if __name__ == '__main__':
    main()
