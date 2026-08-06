import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

from zfa_rl_agent.core.agent.PPO.rPPO import PPO
from zfa_rl_agent.core.environments.grating import swimmer_grating
from zfa_rl_agent.core.environments.zebrafish import swimmer


def build_env(env_name: str, height: int = 256, width: int = 256):
    if env_name == "drift":
        return swimmer(
            force_magnitude=0.0,
            view_render_args={"height": height, "width": width, "camera_id": 0},
        )
    if env_name == "grating":
        return swimmer_grating(
            grating_speed=0.01,
            view_render_args={"height": height, "width": width, "camera_id": 0},
        )
    raise ValueError(f"Unknown environment '{env_name}'. Expected 'drift' or 'grating'.")


def main():
    parser = argparse.ArgumentParser(description="Render a trained policy in the MuJoCo swimmer environment")
    parser.add_argument("--model-path", required=True, help="Path to the trained SB3 PPO zip file")
    parser.add_argument("--env", choices=["drift", "grating"], required=True)
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to render")
    parser.add_argument("--output-dir", default="./render_output", help="Directory for saved frames")
    parser.add_argument("--device", default="cpu", help="Torch device to use (cpu or cuda)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.env, height=args.height, width=args.width)
    model = PPO.load(args.model_path, env=env, device=args.device)

    obs, info = env.reset()
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            Image.fromarray(np.asarray(frame)).save(output_dir / f"frame_{step:03d}.png")
        print(f"step={step:03d} reward={reward:.3f} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"Saved frames to {output_dir}")


if __name__ == "__main__":
    main()
