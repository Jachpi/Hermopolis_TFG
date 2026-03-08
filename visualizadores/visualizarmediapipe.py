import json
import numpy as np
import open3d as o3d
import time
import os

JSON_PATH = "pose_graph_sequence.json"
FRAME_SPEED = 30

if not os.path.isfile(JSON_PATH):
    print("Archivo JSON no encontrado.")
    exit()

# ==============================
# Cargar JSON
# ==============================
with open(JSON_PATH, "r") as f:
    graph_sequence = json.load(f)

# Ordenar por frame_index por seguridad
graph_sequence = sorted(graph_sequence, key=lambda x: x["frame_index"])

print("Frames totales:", len(graph_sequence))

# ==============================
# Obtener lista fija de nodos
# ==============================
# Tomamos los nodos del primer frame como referencia
first_nodes = graph_sequence[0]["nodes"]

# IMPORTANTE: convertir claves a int y ordenarlas
NODE_IDS = sorted([int(k) for k in first_nodes.keys()])

# Crear mapa node_id -> índice consecutivo
index_map = {node_id: idx for idx, node_id in enumerate(NODE_IDS)}

print("Joints finales:", len(NODE_IDS))

# ==============================
# Construir array (frames, joints, 3)
# ==============================
data = []

for frame_data in graph_sequence:
    nodes = frame_data["nodes"]
    frame_points = []

    for node_id in NODE_IDS:
        node = nodes[str(node_id)]  # las claves en JSON son string
        frame_points.append([node["x"], -node["y"], node["z"]])

    data.append(frame_points)

data = np.array(data)

# ==============================
# Obtener aristas y convertir índices
# ==============================
EDGES_OLD = graph_sequence[0]["edges"]
EDGES = [(index_map[a], index_map[b]) for a, b in EDGES_OLD]

# ==============================
# Suavizado opcional
# ==============================
def smooth(data, window=5):
    smoothed = np.copy(data)
    for i in range(window, len(data)):
        smoothed[i] = np.mean(data[i-window:i], axis=0)
    return smoothed

data = smooth(data, window=8)

# ==============================
# Visualización
# ==============================
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_view_control().set_front([1, 0, -1])

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