import numpy as np


def compute_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the face normal using the cross product."""
    normal = np.cross(v1 - v0, v2 - v0)
    return normal / np.linalg.norm(normal)


def rotate_y(theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])


def rotate_z(theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])


def rotate_x(theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])


def barycentric(a, b, c, p):
    s = np.array([
        [c[0] - a[0], b[0] - a[0], a[0] - p[0]],
        [c[1] - a[1], b[1] - a[1], a[1] - p[1]]
    ])
    u = np.cross(s[0], s[1])
    if abs(u[2]) < 1e-2:  # Triangle is degenerate
        return -1, -1, -1
    return 1.0 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]