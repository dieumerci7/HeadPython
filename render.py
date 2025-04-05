import argparse
from typing import Tuple
import numpy as np
import cv2
import imageio.v3 as iio
from model import Model
from utils import compute_normal, rotate_x, rotate_y, rotate_z, barycentric


def render_model(model: Model, texture: np.ndarray, angles: Tuple[float, float, float], width: int = 800, height: int = 800) -> np.ndarray:
    theta_x, theta_y, theta_z = angles
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), -np.inf)
    light_dir = np.array([0, 0, 1])

    tex_h, tex_w = texture.shape[:2]

    # Apply rotation around all axes
    rotation_matrix = rotate_z(theta_z) @ rotate_y(theta_y) @ rotate_x(theta_x)

    for i in range(model.nfaces()):
        face = model.face(i)
        v0, v1, v2 = model.vert(face[0]), model.vert(face[1]), model.vert(face[2])
        # Get texture coordinates
        tex_face = model.tex_face(i)
        vt0 = model.texcoord(tex_face[0])
        vt1 = model.texcoord(tex_face[1])
        vt2 = model.texcoord(tex_face[2])

        # Apply rotation
        v0, v1, v2 = rotation_matrix @ v0, rotation_matrix @ v1, rotation_matrix @ v2

        # Compute shading
        normal = compute_normal(v0, v1, v2)
        intensity = np.dot(normal, light_dir)
        if intensity <= 0:
            continue  # Skip back-facing triangles

        # Convert 3D to 2D screen coordinates
        pts = np.array([
            [(v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2],
            [(v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2],
            [(v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2]
        ], dtype=np.int32)

        z_avg = (v0[2] + v1[2] + v2[2]) / 3

        triangle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, pts, 255)

        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width - 1, max_x), min(height - 1, max_y)

        # Update Z-buffer and image
        for y in range(int(min_y), int(max_y)):
            for x in range(int(min_x), int(max_x)):
                if triangle_mask[y, x] == 255:
                    # Compute barycentric coordinates
                    u, v, w = barycentric(pts[0], pts[1], pts[2], (x, y))
                    if u < 0 or v < 0 or w < 0:
                        continue  # Outside triangle or degenerate

                    if u >= 0 and v >= 0 and w >= 0:
                        z_pixel = u * v0[2] + v * v1[2] + w * v2[2]
                        if z_pixel > z_buffer[y, x]:
                            # Interpolate texture coordinates
                            tex_coord = u * vt0 + v * vt1 + w * vt2
                            tx, ty = int(tex_coord[0] * tex_w), int((1 - tex_coord[1]) * tex_h)

                            # Ensure within texture bounds
                            tx = np.clip(tx, 0, tex_w - 1)
                            ty = np.clip(ty, 0, tex_h - 1)

                            # Sample texture color
                            texture_color = texture[ty, tx]

                            # Apply shading
                            shaded_color = (texture_color * intensity).astype(np.uint8)

                            # Update pixel
                            z_buffer[y, x] = z_pixel
                            image[y, x] = shaded_color

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a rotating 3D head.")
    parser.add_argument("--roll", type=float, default=0, help="Angle of rotation around X-axis in degrees, per frame.")
    parser.add_argument("--pitch", type=float, default=0, help="Angle of rotation around Y-axis in degrees, per frame.")
    parser.add_argument("--yaw", type=float, default=0, help="Angle of rotation around Z-axis in degrees, per frame.")
    args = parser.parse_args()

    model = Model("data/african_head.obj")
    texture = iio.imread("data/african_head_diffuse.tga")

    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    tex_h, tex_w = texture.shape[:2]
    theta_x = theta_y = theta_z = 0  # Initial rotation angles

    while True:
        image = render_model(model, texture, (theta_x, theta_y, theta_z))
        cv2.imshow("Rotating Head", image)

        theta_x += np.radians(args.roll)
        theta_y += np.radians(args.pitch)
        theta_z += np.radians(args.yaw)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()