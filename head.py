import cv2
import numpy as np

class Model:
    def __init__(self, filename):
        self.verts = []
        self.normals = []
        self.faces = []
        
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):  # Vertex positions
                    parts = line.strip().split()
                    self.verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32))
                elif line.startswith("f "):  # Faces
                    parts = line.strip().split()
                    face = []
                    for p in parts[1:]:
                        v_idx, _, _ = (p.split("/") + ["-1", "-1"])[:3]
                        face.append(int(v_idx) - 1)
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


def render_model(model, width=800, height=800):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), -np.inf)  # To track depth
    light_dir = np.array([0, 0, 1])  # Light coming from the front
    
    for i in range(model.nfaces()):
        face = model.face(i)
        v0, v1, v2 = model.vert(face[0]), model.vert(face[1]), model.vert(face[2])

        # Compute face normal
        normal = compute_normal(v0, v1, v2)
        
        normal /= np.linalg.norm(normal)
        
        # Compute intensity using the dot product with the light direction
        intensity = np.dot(normal, light_dir)
        if intensity <= 0:
            continue  # Skip faces with no lighting

        color = (intensity * 255, intensity * 255, intensity * 255)

        # Convert 3D to 2D screen space
        pts = np.array([
            [(v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2],
            [(v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2],
            [(v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2]
        ], dtype=np.int32)

        # Compute average Z depth for this triangle
        z_avg = (v0[2] + v1[2] + v2[2]) / 3

        triangle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, pts, 255)

        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width - 1, max_x), min(height - 1, max_y)

        # Update Z-buffer and image where the mask is filled
        for y in range(int(min_y), int(max_y)):
            for x in range(int(min_x), int(max_x)):
                if triangle_mask[y, x] == 255:  # If inside the triangle
                    if z_avg > z_buffer[y, x]:  # Depth test
                        z_buffer[y, x] = z_avg  # Update depth
                        image[y, x] = color     # Update color
    return image


if __name__ == "__main__":
    model = Model("african_head.obj")
    image = render_model(model)
    
    cv2.imshow("Shading Model", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
