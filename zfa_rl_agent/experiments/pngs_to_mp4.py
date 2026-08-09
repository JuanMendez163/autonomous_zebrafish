import argparse
from pathlib import Path

import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser(description="Convert a folder of PNG frames into an MP4 video")
    parser.add_argument("--input-dir", required=True, help="Directory containing frame_*.png files")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=20.0, help="Frames per second for the output video")
    parser.add_argument("--sort", action="store_true", help="Sort frames numerically instead of lexically")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    if args.sort:
        frames = sorted(frames, key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else p.name)
    else:
        frames = sorted(frames, key=lambda p: p.name)

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", format="FFMPEG")
    try:
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()

    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
