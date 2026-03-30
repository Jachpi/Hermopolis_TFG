import argparse
import re
import torch
import normalizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--type", required=True)
parser.add_argument("--wPadding", action="store_true")
parser.add_argument("--taichiType")
args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT = args.output
INPUT_FLAG = args.type # Sólo puede ser o --type mediapipe o bien --type kinect

# Se guardará en json
if INPUT_FLAG == "mediapipe":
    if not args.taichiType:
        raise Exception("Para usar la transformación desde mediapipe, se debe usar --taichiType y definir el tipo de movimiento.")
    from mediapipe_transformer import PoseGraphExtractor
    import to_kinect_converter
    poseG = PoseGraphExtractor("pose_landmarker_heavy.task")
    initial_data = poseG.process_video(INPUT_PATH)
    kinectizised_data = to_kinect_converter.convert_from_mediapipe(initial_data)
    normalized_data = normalizer.normalize(kinectizised_data)
    tensor = normalizer.to_st_gcn(normalized_data)

    GESTURE_INDEX = args.taichiType

    normalizer.save_json(normalized_data, OUTPUT)

# Se recomienda guardar en .pt
elif INPUT_FLAG == "kinect":
    import kinect_transformer

    match = re.search(r"G\d\d", INPUT_PATH)
    if not match:
        print("No se encontró etiqueta de gesto en el nombre del archivo")
        exit()
    gesture_label = match.group()
    gesture_index = int(gesture_label[1:]) - 1
    print(gesture_index)

    json_kinect = kinect_transformer.kinect_transform(INPUT_PATH)
    normalized_data = normalizer.normalize(json_kinect)
    if args.wPadding:
        T_MAX = normalizer.T_MAX_B if gesture_index == 12 else normalizer.T_MAX_A
    else:
        T_MAX = None

    tensor = normalizer.to_st_gcn(normalized_data, T_MAX)

    # Guardar tensor + etiqueta juntos
    torch.save({"data": tensor, "label": gesture_index}, OUTPUT)
    print(f"Gesto: {gesture_label} (índice {gesture_index}) | Tensor shape: {tensor.shape}")