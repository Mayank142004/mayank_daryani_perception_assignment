import argparse
from scripts.run_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help="Scene name")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    
    run_pipeline(args.scene, 'config.yaml', 'outputs', args.max_frames)

if __name__ == '__main__':
    main()
