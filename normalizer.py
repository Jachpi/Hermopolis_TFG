import os
import json
import math


INPUT_PATH = "pose_graph_translated.json"

OUTPUT_PATH = "final_data.json"

    
class Normalizer:
    def __init__(self):
        self.KEEP = [
            0,3,4,5,6,8,9,10,
            12,13,14,15,
            16,17,18,19,
            21,22,23,24
        ]
    def normalize(self, in_file, out_file):
        if not os.path.isfile(in_file):
            print("Archivo no encontrado")
            exit()
        with open(in_file, "r") as f:
            media_dist = 0
            data = json.load(f)
            spinebase = data[0]["nodes"]["0"]
            head = data[0]["nodes"]["3"]
            factor_divisor = {
                "x": float(spinebase["x"])-float(head["x"]),
                "y": float(spinebase["y"])-float(head["y"]),
                "z": float(spinebase["z"])-float(head["z"]),
            }
            distancia_factor = math.sqrt(
                factor_divisor["x"]**2+
                factor_divisor["y"]**2+
                factor_divisor["z"]**2
            )
            print(distancia_factor)
            for frame in data:
                for node in frame["nodes"].values():
                    node["x"] = float(node["x"])/distancia_factor
                    node["y"] = float(node["y"])/distancia_factor
                    node["z"] = float(node["z"])/distancia_factor
            with open(out_file, "w") as j:
                json.dump(data,j,indent=2)

if __name__ == "__main__":
    if not os.path.isfile(INPUT_PATH):
        print("Archivo no encontrado")
        exit()
    nom = Normalizer()
    nom.normalize(INPUT_PATH,OUTPUT_PATH)
