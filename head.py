import cv2
import numpy as np

class Model:
    def __init__(self, filename):
        self.verts = []
        self.faces = []
        
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):  # Vertex positions
                    parts = line.strip().split()
                    self.verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32))
                elif line.startswith("f "):  # Faces
                    parts = line.strip().split()
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    self.faces.append(face)

    def vert(self, i):
        return self.verts[i]

    def face(self, i):
        return self.faces[i]

    def nfaces(self):
        return len(self.faces)


def compute_normal(v0, v1, v2):
    """Compute the face normal using the cross product."""
    normal = np.cross(v1 - v0, v2 - v0)
    return normal / np.linalg.norm(normal)


def rotate_x(theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])


def rotate_y(theta):
    """Rotation matrix for Y-axis."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])


def rotate_z(theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])


def render_model(model, angles, width=800, height=800):
    theta_x, theta_y, theta_z = angles
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), -np.inf)
    light_dir = np.array([0, 0, 1])

    # Apply rotation around all axes
    rotation_matrix = rotate_z(theta_z) @ rotate_y(theta_y) @ rotate_x(theta_x)

    for i in range(model.nfaces()):
        face = model.face(i)
        v0, v1, v2 = model.vert(face[0]), model.vert(face[1]), model.vert(face[2])

        # Apply rotation
        v0, v1, v2 = rotation_matrix @ v0, rotation_matrix @ v1, rotation_matrix @ v2

        normal = compute_normal(v0, v1, v2)
        intensity = np.dot(normal, light_dir)

        if intensity <= 0:
            continue  # Skip back-facing triangles

        color = (intensity * 255, intensity * 255, intensity * 255)

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
                    if z_avg > z_buffer[y, x]:
                        z_buffer[y, x] = z_avg
                        image[y, x] = color
    return image


if __name__ == "__main__":
    model = Model("african_head.obj")
    theta_x = theta_y = theta_z = 0  # Initial rotation angles

    while True:
        image = render_model(model, (theta_x, theta_y, theta_z))
        cv2.imshow("Rotating Head", image)
        # Rotate 2˚ per frame
        theta_x += np.radians(2)
        theta_y += np.radians(2)
        theta_z += np.radians(2)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()
