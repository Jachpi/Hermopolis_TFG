import json

KINECTNODES = {
    "SPINEBASE": 0,
    "SPINEMID": 1,
    "NECK": 2,
    "HEAD": 3,
    "SHOULDERLEFT": 4,
    "ELBOWLEFT": 5,
    "WRISTLEFT": 6,
    "HANDLEFT": 7,
    "SHOULDERRIGHT": 8,
    "ELBOWRIGHT": 9,
    "WRISTRIGHT": 10,
    "HANDRIGHT": 11,
    "HIPLEFT": 12,
    "KNEELEFT": 13,
    "ANKLELEFT": 14,
    "FOOTLEFT": 15,
    "HIPRIGHT": 16,
    "KNEERIGHT": 17,
    "ANKLERIGHT": 18,
    "FOOTRIGHT": 19,
    "SPINESHOULDER": 20,
    "HANDTIPLEFT": 21,
    "THUMBLEFT": 22,
    "HANDTIPRIGHT": 23,
    "THUMBRIGHT": 24
}

NODETRANSLATIONS = {
    34: "SPINEBASE",
    33: "HEAD",
    11: "SHOULDERLEFT",
    13: "ELBOWLEFT",
    15: "WRISTLEFT",
    12: "SHOULDERRIGHT",
    14: "ELBOWRIGHT",
    16: "WRISTRIGHT",
    23: "HIPLEFT",
    25: "KNEELEFT",
    27: "ANKLELEFT",
    31: "FOOTLEFT",
    24: "HIPRIGHT",
    26: "KNEERIGHT",
    28: "ANKLERIGHT",
    32: "FOOTRIGHT",
    19: "HANDTIPLEFT",
    21: "THUMBLEFT",
    20: "HANDTIPRIGHT",
    22: "THUMBRIGHT"
}

def convert_from_mediapipe(data):
    '''Convierte una secuencia de poses de MediaPipe al formato de índices Kinect.

    Traduce las claves de nodo de índices BlazePose (p.ej. 11, 23, 33) a índices
    enteros del esqueleto Kinect (p.ej. 4, 12, 0) usando NODETRANSLATIONS y
    KINECTNODES. Adicionalmente invierte el eje Y para alinear la orientación con
    el sistema de coordenadas de Kinect (Y hacia arriba).

    Args:
        data (list[dict]): Secuencia de frames producida por
            PoseGraphExtractor.process_video, con nodos indexados por IDs BlazePose.

    Returns:
        list[dict]: Secuencia de frames con nodos indexados por IDs Kinect (int)
            y eje Y invertido. Conserva frame_index y timestamp_ms originales.
    '''
    translated_data = []
    for frame in data:
        new_frame = {"frame_index": frame["frame_index"], "timestamp_ms": frame["timestamp_ms"], "nodes": {}}
        for node in frame["nodes"]:
            new_frame["nodes"][KINECTNODES[NODETRANSLATIONS[int(node)]]] = frame["nodes"][node]
            new_frame["nodes"][KINECTNODES[NODETRANSLATIONS[int(node)]]]["y"] *= -1
        translated_data.append(new_frame)
    return translated_data

def save_json(data, output_path):
    '''Serializa los datos traducidos a formato Kinect en un archivo JSON.

    Args:
        data (list[dict]): Secuencia de frames con índices Kinect.
        output_path (str): Ruta del archivo JSON de salida.
    '''
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Datos traducidos guardados en: {output_path}")

if __name__ == "__main__":
    FILE_INPUT = "medianodes.json"
    OUTPUT_JSON = "synchronized.json"

    with open(FILE_INPUT, "r") as d:
        data = json.load(d)

    translated = convert_from_mediapipe(data)
    save_json(translated, OUTPUT_JSON)