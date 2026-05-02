import os
import json

KEEP = [
    0,3,4,5,6,8,9,10,
    12,13,14,15,
    16,17,18,19,
    21,22,23,24
]

def kinect_transform(file):
    if not os.path.isfile(file):
        print("Archivo no encontrado")
        return None

    json_data = []
    with open(file, "r") as f:
        f_it = 0
        timestamp = 0
        first_line = f.readline().strip().split()
        first_timestamp = int(first_line[0])
        f.seek(0)
        # Leer línea
        for line in f:
            data = line.strip().split()
            timestamp += int(data[0]) - first_timestamp
            frame = {"frame_index": f_it, "timestamp_ms": timestamp}
            exit = False
            it = 1 # Comienza en 1 porque data[0] es el timestamp
            nodes = {}
            while not exit:
                if it > 73:
                    exit = True
                else:
                    nodes[int(it/3)] = {
                        "x": (float(data[it+1])),
                        "y": (float(data[it+2])),
                        "z": (float(data[it])),
                        "visibility": 1.0
                    }
                it += 3
            frame["nodes"] = nodes
            json_data.append(frame)
            f_it += 1
    return json_data

def save_json(data, output_path):
    with open(output_path, "w") as j:
        json.dump(data, j, indent=2)
    print(f"Datos Kinect guardados en: {output_path}")

if __name__ == "__main__":
    FILE_PATH = "../otrosArchivos/P01T01C03G01D01S01.txt"
    OUTPUT_PATH = "kinectnodes.json"

    data = kinect_transform(FILE_PATH)
    if data:
        save_json(data, OUTPUT_PATH)