"""Command-line entry point: python -m schrodinger_pitch"""
import argparse

from .config import DEFAULT_FRAME, DEFAULT_MATCH_ID, DEFAULT_SEED
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run the Schrödinger Pitch Control analysis.")
    parser.add_argument("--match-id", type=int, default=DEFAULT_MATCH_ID)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-draws", type=int, default=64, help="ghost draws per frame or pass")
    parser.add_argument("--n-frames", type=int, default=200, help="frames in the match-wide sample")
    parser.add_argument("--data-dir", default=None, help="folder holding matches/<id>/")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.match_id, args.frame, args.seed, args.n_draws, args.n_frames, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
