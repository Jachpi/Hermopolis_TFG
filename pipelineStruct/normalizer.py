import json
import math
import numpy as np
import torch

KEEP = [
    0,3,4,5,6,8,9,10,
    12,13,14,15,
    16,17,18,19,
    21,22,23,24
]

def normalize(data):
    '''Normaliza la secuencia de poses centrando en SPINEBASE y escalando por la distancia SPINEBASE-HEAD.

    La distancia euclidiana entre SPINEBASE (nodo 0) y HEAD (nodo 3) del primer
    frame se usa como factor de escala global. En cada frame todos los nodos se
    trasladan para que SPINEBASE quede en el origen (0, 0, 0) y se dividen por
    dicho factor. Soporta claves de nodo tanto int como str.

    Args:
        data (list[dict]): Secuencia de frames con nodos en coordenadas absolutas.
            Compatible con la salida de kinect_transformer, mmpose_transformer y
            to_kinect_converter.

    Returns:
        list[dict]: La misma lista modificada en sitio con coordenadas normalizadas.
    '''
    # Calcular factor de escala a partir del primer frame (distancia spinebase->head)

    k0 = type(next(iter(data[0]["nodes"])))(0) # El json de nodos puede tener keys en formato string o int. Esto elimina esta posible ambigüedad.
    k3 = type(next(iter(data[0]["nodes"])))(3)

    spinebase = data[0]["nodes"][k0]
    head = data[0]["nodes"][k3]
    factor_divisor = {
        "x": float(spinebase["x"]) - float(head["x"]),
        "y": float(spinebase["y"]) - float(head["y"]),
        "z": float(spinebase["z"]) - float(head["z"]),
    }
    distancia_factor = math.sqrt(
        factor_divisor["x"]**2 +
        factor_divisor["y"]**2 +
        factor_divisor["z"]**2
    )
    print(f"Factor de escala (distancia spinebase->head): {distancia_factor}")

    for frame in data:
        # Capturar spinebase antes de modificar ningún nodo
        spb_x = float(frame["nodes"][k0]["x"])
        spb_y = float(frame["nodes"][k0]["y"])
        spb_z = float(frame["nodes"][k0]["z"])
        for node in frame["nodes"].values():
            node["x"] = (float(node["x"]) - spb_x) / distancia_factor
            node["y"] = (float(node["y"]) - spb_y) / distancia_factor
            node["z"] = (float(node["z"]) - spb_z) / distancia_factor

    return data

def to_st_gcn(data, t_max=None):
    '''Convierte una secuencia de frames normalizada a un tensor ST-GCN.

    Selecciona los nodos definidos en KEEP y los organiza en un tensor de cuatro
    dimensiones compatible con el formato de entrada de PoseDataset de MMAction2.
    Si t_max es mayor que len(data), los frames sobrantes se rellenan con ceros
    (padding temporal).

    Args:
        data (list[dict]): Secuencia de frames normalizada (salida de normalize).
        t_max (int, optional): Longitud temporal fija. Si es None, T = len(data).

    Returns:
        torch.Tensor: Tensor de forma (C=3, T, V=20, M=1) con dtype float32,
            donde C son los ejes x/y/z, T los frames, V los nodos y M=1 el número de personas.
    '''
    if not t_max:
        T = len(data)
    else:
        T = t_max
        
    V = len(KEEP)
    C = 3

    key_type = type(next(iter(data[0]["nodes"])))
    
    stgcn_data = np.zeros((C, T, V)) # t_max hará padding. Si no, len(data) dejará el clip tal cual.

    for t, frame in enumerate(data):
        for n, node_id in enumerate(KEEP):
            node = frame["nodes"][key_type(node_id)]
            stgcn_data[0,t,n] = node["x"]
            stgcn_data[1,t,n] = node["y"]
            stgcn_data[2,t,n] = node["z"]

    tensor = torch.tensor(stgcn_data,dtype=torch.float32).unsqueeze(-1)
    return tensor

def save_json(data, output_path):
    '''Almacena la secuencia de frames normalizada a un archivo JSON.

    Args:
        data (list[dict]): Secuencia de frames a guardar.
        output_path (str): Ruta del archivo JSON de salida.
    '''
    with open(output_path, "w") as j:
        json.dump(data, j, indent=2)
    print(f"Datos normalizados guardados en: {output_path}")

if __name__ == "__main__":
    INPUT_PATH = "kinectnodes.json"
    OUTPUT_PATH = "final_data.json"

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    normalized = normalize(data)
    save_json(normalized, OUTPUT_PATH)