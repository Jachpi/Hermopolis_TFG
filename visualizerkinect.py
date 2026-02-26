import numpy as np
import open3d as o3d
import time
import os
import json

FILE_PATH = "../Segmented_Kinect/Segmented_Kinect/G08/P01T02C03G08D02S01.txt"
FRAME_SPEED = 30

if not os.path.isfile(FILE_PATH):
    print("Archivo no encontrado.")
    exit()

data = []

# ==============================
# Índices que queremos conservar
# ==============================
KEEP = [
    0,3,4,5,6,8,9,10,
    12,13,14,15,
    16,17,18,19,
    21,22,23,24
]

with open(FILE_PATH, "r") as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        coords = values[1:]
        frame = np.array(coords).reshape(25, 3)

        # Filtrar joints
        frame = frame[KEEP]

        data.append(frame)

data = np.array(data)


print("Frames totales:", data.shape[0])
print("Joints finales:", data.shape[1])

# ==============================
# Crear mapa antiguo -> nuevo índice
# ==============================
index_map = {old:new for new, old in enumerate(KEEP)}

# ==============================
# Definir aristas con índices antiguos
# ==============================
EDGES_OLD = [

    # Hombros
    (4,8),

    # Tronco
    (8,16),
    (4,12),
    (16,12),

    # Brazo izquierdo
    (4,5),
    (5,6),

    # Brazo derecho
    (8,9),
    (9,10),

    # Pierna izquierda
    (12,13),
    (13,14),
    (14,15),

    # Pierna derecha
    (16,17),
    (17,18),
    (18,19),

    # Mano izquierda
    (21,6),
    (22,6),

    # Mano derecha
    (23,10),
    (24,10),
]

# ==============================
# Convertir a nuevos índices
# ==============================
EDGES = [(index_map[a], index_map[b]) for a,b in EDGES_OLD]

# ==============================
# Suavizado
# ==============================
def smooth(data, window=5):
    smoothed = np.copy(data)
    for i in range(window, len(data)):
        smoothed[i] = np.mean(data[i-window:i], axis=0)
    return smoothed



data = smooth(data, window=5)

for f in range(len(data)):
    spine_base = data[f][0].copy()
    print("spine_base ",spine_base)
    for n in range(len(data[f])):
        data[f][n] = data[f][n] - spine_base

print(data)

# ==============================
# Visualización
# ==============================
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_view_control().set_front([1,0,-1])

pcd = o3d.geometry.PointCloud()
lines = o3d.geometry.LineSet()

first_frame = data[0]

pcd.points = o3d.utility.Vector3dVector(first_frame)
lines.points = o3d.utility.Vector3dVector(first_frame)
lines.lines = o3d.utility.Vector2iVector(EDGES)

vis.add_geometry(pcd)
vis.add_geometry(lines)

for frame in data:
    if not np.isnan(frame).any():
        pcd.points = o3d.utility.Vector3dVector(frame)
        lines.points = o3d.utility.Vector3dVector(frame)

        vis.update_geometry(pcd)
        vis.update_geometry(lines)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(FRAME_SPEED / 1000.0)

vis.destroy_window()
