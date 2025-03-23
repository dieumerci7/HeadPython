from pathlib import Path
from typing import Union
import numpy as np


class Model:
    def __init__(self, filename: Union[str, Path]):
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

    def vert(self, i: int) -> np.ndarray:
        return self.verts[i]

    def face(self, i: int) -> np.ndarray:
        return self.faces[i]

    def nfaces(self) -> int:
        return len(self.faces)