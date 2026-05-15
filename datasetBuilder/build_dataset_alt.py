'''Construye taichi_dataset_loso.pkl usando la estrategia Leave-One-Subject-Out.

Lee todos los archivos .txt de capturas Kinect de INPUT_PATH y asigna el split
train/val en función del identificador de persona (PXX) extraído del nombre de
archivo. Los sujetos P01 y P12 van a validación; el resto a entrenamiento.
Lanza una excepción si algún archivo no contiene el patrón PXX.

El PKL resultante tiene el mismo formato que build_dataset.py y es compatible
con PoseDataset de MMAction2:
    {
        "split": {"train": [frame_dir, ...], "val": [frame_dir, ...]},
        "annotations": [{"keypoint", "label", "total_frames", "frame_dir"}, ...]
    }
'''
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
OUTPUT_PKL = "../Segmented_Kinect_pkl/taichi_dataset_loso.pkl"  # archivo de salida con todos los clips

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
        "keypoint": tensor.permute(3, 1, 2, 0).numpy(),  # (C,T,V,M) -> (M,T,V,C)
        "label": gesture_index,
        "total_frames": len(normalized_data),
        "frame_dir": f  # nombre del archivo
    }
    annotations.append(sample)
    dataset["annotations"] = annotations
    match2 = re.search(r"P\d\d",f)
    if match2:
        person_label = match2.group()
        person_index = int(person_label[1:])
        if person_index == 1 or person_index == 12:
            dataset["split"]['val'].append(f)
        else:
            dataset["split"]['train'].append(f)
    else:
        raise Exception(f"El archivo {file_path} no contiene una definición de la persona con denotación 'P\d\d' en su nombre de archivo.")

    print(f"Persona: {person_index} Gesto: {gesture_label} (índice {gesture_index}) | Tensor shape: {tensor.shape}")

with open(OUTPUT_PKL, "wb") as out:
    pickle.dump(dataset, out)

print(f"\nDataset guardado en {OUTPUT_PKL} con {len(annotations)} clips")