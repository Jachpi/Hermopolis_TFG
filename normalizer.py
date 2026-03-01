import json
import math

KEEP = [
    0,3,4,5,6,8,9,10,
    12,13,14,15,
    16,17,18,19,
    21,22,23,24
]

def normalize(data):
    # Calcular factor de escala a partir del primer frame (distancia spinebase→head)
    spinebase = data[0]["nodes"][0]
    head = data[0]["nodes"][3]
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
    print(f"Factor de escala (distancia spinebase→head): {distancia_factor}")

    for frame in data:
        # Capturar spinebase antes de modificar ningún nodo
        spb_x = float(frame["nodes"][0]["x"])
        spb_y = float(frame["nodes"][0]["y"])
        spb_z = float(frame["nodes"][0]["z"])
        for node in frame["nodes"].values():
            node["x"] = (float(node["x"]) - spb_x) / distancia_factor
            node["y"] = (float(node["y"]) - spb_y) / distancia_factor
            node["z"] = (float(node["z"]) - spb_z) / distancia_factor

    return data

def save_json(data, output_path):
    with open(output_path, "w") as j:
        json.dump(data, j, indent=2)
    print(f"Datos normalizados guardados en: {output_path}")

if __name__ == "__main__":
    INPUT_PATH = "synchronized.json"
    OUTPUT_PATH = "final_data.json"

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    normalized = normalize(data)
    save_json(normalized, OUTPUT_PATH)