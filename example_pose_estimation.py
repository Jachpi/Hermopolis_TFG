import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = "clip.mp4"
MODEL_PATH = "pose_landmarker_heavy.task"
'''El .task contiene:
- Un modelo preentrenado .tflite
- Metadatos
- Variables para configurar preprocesamiento
- Variables para configurar postprocesamiento
- Información del modelo usado
'''
OUTPUT_JSON = "pose_graph_sequence.json"
OUTPUT_VIDEO = "pose_graph_overlay.mp4"


class PoseGraphExtractor:

    def __init__(self, model_path):

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode #IMAGE, VIDEO o LIVE_STREAM

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False
        )
        '''Las máscaras de segmentación es una img donde cada píxel se le proporciona una
            probabilidad de que pertenece a una clase. En este caso, no nos interesa definir los
            píxeles que pertenecen a, por ejemplo, una persona y los que pertenecen al fondo.
            Reducimos memoria con esto desactivado.'''

        self.landmarker = PoseLandmarker.create_from_options(options)
        '''Inicializa TFlite (Tensorflow Lite) y el grafo a generarse. Usamos directamente un archivo
        .tflite de Google Tensorflow ya entrenado.'''
        self.graph_sequence = []

        self.keep = [
            0,
            11,12,
            13,14,
            15,16,
            19,20,
            21,22,
            23,24,
            25,26,
            27,28,
            31,32
        ]

        # Para calcular el centro craneal después (ojo izq, ojo der, oreja izq, oreja der)
        self.aux_nodes = [2,5,7,8]

        # Conectividad estándar BlazePose (33 landmarks)
        self.edges = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 12),
            (11, 23), (12, 24),
            (23, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
            (27, 31), (28, 32)
        ]

        self.edges = [(i, j) for (i, j) in self.edges if i in self.keep and j in self.keep]

        '''BlazePose cuenta con múltiples nodos. Aquí ponemos las aristas que queremos mostrar
        y detectar. Ej: 11 -> hombo izquierdo; 13 -> codo izquierdo. "(11,13)" Genera una arista entre
        esos dos nodos.
        
        Más info: 
        https://chromium.googlesource.com/chromium/src/%2B/HEAD/third_party/mediapipe/src/mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt

        '''
    def find_head_center(self, landmarks):
        left_eye = landmarks[2]
        right_eye = landmarks[5]        
        eye_mediatriz = np.array([(left_eye.x + right_eye.x)/2, (left_eye.y + right_eye.y)/2, (left_eye.z + right_eye.z)/2])

        left_ear = landmarks[7]
        right_ear = landmarks[8]
        ear_mediatriz = np.array([(left_ear.x + right_ear.x)/2, (left_ear.y + right_ear.y)/2, (left_ear.z + right_ear.z)/2])

        # TODO: CONTINUAR LA APROXIMACIÓN DEL CENTRO DE LA CABEZA





    def draw_graph(self, frame, landmarks):

        '''Dibuja los nodos y aristas definidos.'''

        h, w, _ = frame.shape

        # Dibujar aristas
        for (i, j) in self.edges:
            if i < len(landmarks) and j < len(landmarks):
                xi = int(landmarks[i].x * w)
                yi = int(landmarks[i].y * h)
                xj = int(landmarks[j].x * w)
                yj = int(landmarks[j].y * h)

                cv2.line(frame, (xi, yi), (xj, yj), (0, 255, 0), 2)

        # Dibujar nodos
        for i in self.keep:
            lm = landmarks[i]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x,y), 1, (0, 0, 255), -1)

        return frame

    def process_video(self, video_path, output_video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Para guardar cada frame añadiendo luego el grafo por encima. Es opcional
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0

        exit = False

        while not exit:
            ret, frame = cap.read()
            if not ret:
                exit = True
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Mediapipe usa imágenes RGB, OpenCV usa BGR. Por suerte, existe esta función para convertir entre ambos formatos.
    
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                ) # El formato de imagen que quiere Mediapipe. Es un wrapper de OpenCV.

                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    # Guardar grafo
                    nodes = {}
                    for i in self.keep:
                        lm = landmarks[i]
                        nodes[i] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        }

                    graph_data = {
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "nodes": nodes,
                        "edges": self.edges
                    }

                    self.graph_sequence.append(graph_data)

                    # Dibujar grafo sobre frame
                    frame = self.draw_graph(frame, landmarks)

                out.write(frame)
                frame_idx += 1

        cap.release()
        out.release()

    def save_json(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.graph_sequence, f, indent=2)


if __name__ == "__main__":

    extractor = PoseGraphExtractor(MODEL_PATH)
    extractor.process_video(VIDEO_PATH, OUTPUT_VIDEO)
    extractor.save_json(OUTPUT_JSON)

    print("Proceso completado.")
    print(f"Grafo temporal guardado en: {OUTPUT_JSON}")
    print(f"Vídeo con overlay guardado en: {OUTPUT_VIDEO}")
