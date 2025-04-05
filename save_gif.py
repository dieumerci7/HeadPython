import argparse
import numpy as np
import cv2
import imageio
import imageio.v3 as iio
from model import Model
from render import render_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a rotating 3D head.")
    parser.add_argument("--roll", type=float, default=0, help="X-axis rotation per frame (degrees).")
    parser.add_argument("--pitch", type=float, default=0, help="Y-axis rotation per frame (degrees).")
    parser.add_argument("--yaw", type=float, default=5, help="Z-axis rotation per frame (degrees).")  # Default 5° rotation
    parser.add_argument("--target", type=str, default="data/rotation.gif", help="Output GIF filename.")
    args = parser.parse_args()

    model = Model("data/african_head.obj")
    texture = iio.imread("data/african_head_diffuse.tga")
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    theta_x = theta_y = theta_z = 0  
    frames = []

    for _ in range(72):  # Save 72 frames (~360 degrees if rotating 5° per frame)
        image = render_model(model, texture, (theta_x, theta_y, theta_z))
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB for GIF
        theta_x += np.radians(args.roll)
        theta_y += np.radians(args.pitch)
        theta_z += np.radians(args.yaw)

    imageio.mimsave(args.target, frames, duration=0.1)  # Save GIF

    cv2.destroyAllWindows()