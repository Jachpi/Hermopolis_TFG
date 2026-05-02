import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelineStruct'))

import kinect_transformer
import normalizer
import pickle
import numpy as np
import random
import re

INPUT_PATH = "../Segmented_Kinect"  # ruta hasta el directorio con el dataset de Kinect (con los txt)
OUTPUT_PKL = "../Segmented_Kinect_pkl/taichi_dataset.pkl"  # archivo de salida con todos los clips

dataset = {
    "split": {
        "train":[],
        "val":[]
    },
    "annotations":[]
}
annotations = []

for f in os.listdir(INPUT_PATH):
    file_path = os.path.join(INPUT_PATH, f)
    match = re.search(r"G\d\d", f)

    if not file_path.endswith(".txt") or not match:
        print("Saltando ", file_path, "...")
        continue

    gesture_label = match.group()
    gesture_index = int(gesture_label[1:]) - 1

    json_kinect = kinect_transformer.kinect_transform(file_path)
    normalized_data = normalizer.normalize(json_kinect)
    tensor = normalizer.to_st_gcn(normalized_data)

    #Nota. Éste es el formato que esperará ST-GCN
    sample = {
        "keypoint": tensor.permute(3, 1, 2, 0).numpy(),  # (C,T,V,M) → (M,T,V,C)
        "label": gesture_index,
        "total_frames": len(normalized_data),  # frames reales sin padding
        "frame_dir": f  # nombre del archivo
    }
    annotations.append(sample)
    dataset["annotations"] = annotations
    res = random.choices(population=["train","val"],weights=[80,20])[0]
    dataset["split"][res].append(f)

    print(f"Gesto: {gesture_label} (índice {gesture_index}) | Tensor shape: {tensor.shape}")

with open(OUTPUT_PKL, "wb") as out:
    pickle.dump(dataset, out)

print(f"\nDataset guardado en {OUTPUT_PKL} con {len(annotations)} clips")