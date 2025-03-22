import cv2
import numpy as np


class Model:
    def __init__(self, filename):
        self.verts = []
        self.faces = []
        
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    self.verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                elif line.startswith("f "):
                    parts = line.strip().split()
                    face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    self.faces.append(face)
    
    def nverts(self):
        return len(self.verts)
    
    def nfaces(self):
        return len(self.faces)
    
    def vert(self, i):
        return self.verts[i]
    
    def face(self, i):
        return self.faces[i]


def render_model(model, width=800, height=800):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(model.nfaces()):
        face = model.face(i)
        for j in range(3):
            v0 = model.vert(face[j])
            v1 = model.vert(face[(j + 1) % 3])
            
            x0 = int((v0[0] + 1) * width / 2)
            y0 = int((v0[1] + 1) * height / 2)
            x1 = int((v1[0] + 1) * width / 2)
            y1 = int((v1[1] + 1) * height / 2)
            
            cv2.line(image, (x0, height - y0), (x1, height - y1), (255, 255, 255), 1, cv2.LINE_AA)
    
    return image


if __name__ == "__main__":
    model = Model("african_head.obj")
    image = render_model(model)
    
    cv2.imshow("Display Window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
