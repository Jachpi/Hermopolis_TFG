import os
import json
import math

FILE_PATH = "../Segmented_Kinect/Segmented_Kinect/G08/P01T02C03G08D02S01.txt"
KEEP = [
    0,3,4,5,6,8,9,10,
    12,13,14,15,
    16,17,18,19,
    21,22,23,24
]

OUTPUT_PATH = "kinectnodes.json"

if not os.path.isfile(FILE_PATH):
    print("Archivo no encontrado")
    exit()

class KinectTransformer:
    def __init__(self):
        self.json_data = []
    def kinect_transform(self, file):
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
                frame = {"frame_index": f_it, "timestamp_ms": timestamp,}
                exit = False
                it = 1 # Comienza en 1 porque data[0] es el timestamp
                nodes = {}
                while not exit:
                    if it > 73:
                        exit = True
                    else:
                        nodes[int(it/3)] = {
                            "x":(float(data[it+1]) - float(data[2])),
                            "y":(float(data[it+2]) - float(data[3])),
                            "z":(float(data[it]) - float(data[1])),
                            "visibility":1.0
                        }
                    it += 3
                frame["nodes"] = nodes
                self.json_data.append(frame)
                f_it += 1
    def save_json(self, out):
        with open(out, "w") as j:
                json.dump(self.json_data, j, indent=2)

if __name__ == "__main__":
    k = KinectTransformer()
    k.kinect_transform(FILE_PATH)
    k.save_json(OUTPUT_PATH)