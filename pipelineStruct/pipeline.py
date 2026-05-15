'''Pipeline de inferencia y preprocesado de gestos Taichi.

Procesa un vídeo o archivo de captura Kinect y produce un tensor ST-GCN listo
para entrenamiento o una predicción de gesto con un checkpoint entrenado.

Modos (--type):
    mediapipe   Extrae la pose con MediaPipe, convierte a Kinect, normaliza y
                clasifica con --wCheckpoint.
    mmpose      Extrae la pose 3D con MMPose/MotionBERT, normaliza y
                clasifica con --wCheckpoint.
    kinect      Lee un archivo .txt de captura Kinect, normaliza y guarda el
                tensor en --output (obligatorio en este modo).

Args CLI:
    --input       Ruta al archivo de entrada (vídeo .mp4 o .txt Kinect).
    --output      Ruta del tensor de salida .pt (obligatorio para --type kinect).
    --type        Modo de extracción: mediapipe | mmpose | kinect.
    --wCheckpoint Ruta a un checkpoint .pth para clasificar el gesto.
    --wPadding    (Solo kinect) Aplica padding temporal al tensor.
'''
import sys
sys.path.insert(0, '..')
import argparse
import re
import torch
import normalizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=False)
parser.add_argument("--type", required=True, choices=["mediapipe", "mmpose", "kinect"])
parser.add_argument("--wPadding", action="store_true")
parser.add_argument("--wCheckpoint", required=False)
args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT = args.output
INPUT_FLAG = args.type

if INPUT_FLAG == "kinect" and not OUTPUT:
    parser.error("--type kinect requiere --output para guardar el tensor resultante")

# Se guardará en json
if INPUT_FLAG == "mediapipe":
    from mediapipe_transformer import PoseGraphExtractor
    import to_kinect_converter
    poseG = PoseGraphExtractor("pose_landmarker_heavy.task")
    initial_data = poseG.process_video(INPUT_PATH)
    kinectizised_data = to_kinect_converter.convert_from_mediapipe(initial_data)
    normalized_data = normalizer.normalize(kinectizised_data)
    tensor = normalizer.to_st_gcn(normalized_data)
    sample = {
        "keypoint": tensor.permute(3, 1, 2, 0).numpy(),  # (C,T,V,M) → (M,T,V,C)
        "label": -1,
        "total_frames": len(normalized_data),
    }
    if args.wCheckpoint:
        from mmaction2.mmaction.apis import init_recognizer, inference_recognizer
        from mmaction2.mmaction.utils import register_all_modules
        register_all_modules()
        GESTURE_NAMES = [
            'G01 - Beginning position (Wuji)',
            'G02 - Tree posture (Taiji)',
            'G03 - Open and close lotus flower',
            'G04 - Bring sky and earth together',
            'G05 - Canalize energy',
            'G06 - Drive the monkey away',
            'G07 - Move hands like clouds',
            'G08 - Part the wild horse\'s mane',
            'G09 - Golden rooster stands on one leg',
            'G10 - Fair lady works shuttles',
            'G11 - Kick with heel',
            'G12 - Brush knee and twist step',
            'G13 - Grasp the bird\'s tail'
        ]

        config_file = '../mmaction2/configs/skeleton/stgcn/stgcn_taichi_jm.py'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = init_recognizer(config_file, args.wCheckpoint, device)
        result = inference_recognizer(model, sample)
        scores = result.pred_score.tolist()
        predicted_index = scores.index(max(scores))
        print(f"Predicción: {GESTURE_NAMES[predicted_index]} (confianza: {max(scores):.4f})")

elif INPUT_FLAG == "mmpose":
    from mmpose_transformer import MMPoseExtractor

    extractor = MMPoseExtractor(smooth_window=5)
    initial_data = extractor.process_video(INPUT_PATH)
    normalized_data = normalizer.normalize(initial_data)
    tensor = normalizer.to_st_gcn(normalized_data)

    if args.wCheckpoint:
        from mmaction2.mmaction.apis import init_recognizer, inference_recognizer
        from mmaction2.mmaction.utils import register_all_modules
        register_all_modules()

        GESTURE_NAMES = [
            'G01 - Beginning position (Wuji)',
            'G02 - Tree posture (Taiji)',
            'G03 - Open and close lotus flower',
            'G04 - Bring sky and earth together',
            'G05 - Canalize energy',
            'G06 - Drive the monkey away',
            'G07 - Move hands like clouds',
            'G08 - Part the wild horse\'s mane',
            'G09 - Golden rooster stands on one leg',
            'G10 - Fair lady works shuttles',
            'G11 - Kick with heel',
            'G12 - Brush knee and twist step',
            'G13 - Grasp the bird\'s tail'
        ]

        config_file = '../mmaction2/configs/skeleton/stgcn/stgcn_taichi_jm.py'
        sample = {
            "keypoint": tensor.permute(3, 1, 2, 0).numpy(),
            "label": -1,
            "total_frames": len(normalized_data),
        }
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = init_recognizer(config_file, args.wCheckpoint, device)
        result = inference_recognizer(model, sample)
        scores = result.pred_score.tolist()
        predicted_index = scores.index(max(scores))
        print(f"Prediccion: {GESTURE_NAMES[predicted_index]} (confianza: {max(scores):.4f})")

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