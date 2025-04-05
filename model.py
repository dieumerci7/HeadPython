from pathlib import Path
from typing import Union
import numpy as np


class Model:
    def __init__(self, filename: Union[str, Path]):
        self.verts = []
        self.texcoords = []  # Store texture coordinates (vt)
        self.faces = []
        self.tex_faces = []  # Store texture indices for faces
        
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":  # Vertex positions
                    self.verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32))
                elif parts[0] == "vt":  # Texture coordinates
                    self.texcoords.append(np.array([float(parts[1]), float(parts[2])], dtype=np.float32))
                elif parts[0] == "f":  # Faces (v/vt/vn)
                    face = []
                    tex_face = []
                    for part in parts[1:]:
                        indices = list(map(int, part.split("/"))) + [0, 0]  # Ensure it has v, vt
                        face.append(indices[0] - 1)  # Vertex index
                        tex_face.append(indices[1] - 1 if indices[1] else 0)  # Texture index (handle missing vt)
                    self.faces.append(face)
                    self.tex_faces.append(tex_face)

    def vert(self, i: int) -> np.ndarray:
        return self.verts[i]

    def texcoord(self, i: int) -> np.ndarray:
        return self.texcoords[i]

    def face(self, i: int) -> np.ndarray:
        return self.faces[i]

    def tex_face(self, i: int) -> np.ndarray:
        return self.tex_faces[i]

    def nfaces(self) -> int:
        return len(self.faces)
