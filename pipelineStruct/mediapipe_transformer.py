import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
        self.HEAD = 33
        self.SPINEBASE = 34
        self.head_node = None
        self.spine_base_node = None

        # Para calcular el centro craneal después (ojo izq, ojo der, oreja izq, oreja der)

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

        left_ear = landmarks[7]
        right_ear = landmarks[8]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        dx1 = right_ear.x - left_ear.x
        dz1 = right_ear.z - left_ear.z

        dx2 = right_shoulder.x - left_shoulder.x
        dz2 = right_shoulder.z - left_shoulder.z

        '''
        tdx1 - udx2 = x3 - x1
        tdz1 - udz2 = z3 - z1

        siendo t y u los parámetros escalares que definen en qué punto de la recta nos encontramos.
        '''

        A = np.array([[dx1,-dx2],[dz1,-dz2]])
        b = np.array([left_shoulder.x - left_ear.x, left_shoulder.z - left_ear.z])

        det = np.linalg.det(A)
        if abs(det) < 0.01:
            # Las rectas generadas son paralelas prácticamente. Aproximar a la mediatriz de las orejas
            return np.array([(left_ear.x+right_ear.x)/2, (left_ear.y+right_ear.y)/2, (left_ear.z+right_ear.z)/2])
        
        t, u = np.linalg.solve(A, b)

        x_final = left_ear.x + t*dx1
        z_final = left_ear.z + t*dz1

        return np.array([x_final, (left_ear.y+right_ear.y)/2, z_final])

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
        # Dibujar cabeza
        if self.head_node is not None:
            x = int(self.head_node[0] * w)
            y = int(self.head_node[1] * h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
        # Dibujar spinebase
        if self.spine_base_node is not None:
            x = int(self.spine_base_node[0]*w)
            y = int(self.spine_base_node[1]*h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        return frame

    def find_spine_base(self, landmarks):
        hip_left = landmarks[23]
        hip_right = landmarks[24]
        return np.array([(hip_left.x+hip_right.x)/2, (hip_left.y+hip_right.y)/2, (hip_left.z+hip_right.z)/2])

    def save_json(self, data, output_path):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Grafo temporal guardado en: {output_path}")

    def process_video(self, video_path, output_video_path=None):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Para guardar cada frame añadiendo luego el grafo por encima. Es opcional
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        exit = False

        while not exit:
            ret, frame = cap.read()
            if not ret:
                exit = True
            else:
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Mediapipe usa imágenes RGB, OpenCV usa BGR. Por suerte, existe esta función para convertir entre ambos formatos.
    
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                ) # El formato de imagen que necesita Mediapipe. Es un wrapper de OpenCV.

                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    # Guardar grafo
                    nodes = {}

                    # Calcular spinebase y guardarlo en los nodos
                    self.spine_base_node = self.find_spine_base(landmarks)
                    nodes[self.SPINEBASE] = {"x":float(self.spine_base_node[0]), "y":float(self.spine_base_node[1]), "z":float(self.spine_base_node[2]),"visibility": 1.0}
                    
                    # Calcular cabeza y guardarlo en los nodos
                    self.head_node = self.find_head_center(landmarks)
                    nodes[self.HEAD] = {"x":float(self.head_node[0]), "y":float(self.head_node[1]), "z":float(self.head_node[2]),"visibility": 1.0}
                    
                    # Guardar el resto de nodos existentes
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

                    if output_video_path:
                        # Dibujar grafo sobre frame
                        frame = self.draw_graph(frame, landmarks)

                if output_video_path:
                    out.write(frame)

                frame_idx += 1

        cap.release()
        if output_video_path:
            out.release()
            print(f"Vídeo con overlay guardado en: {output_video_path}")
        
        self.landmarker.close()
        return self.graph_sequence

if __name__ == "__main__":
    VIDEO_PATH = "../videos/clip2.mp4"
    MODEL_PATH = "pose_landmarker_heavy.task"
    OUTPUT_JSON = "medianodes.json"
    OUTPUT_VIDEO = "pose_graph_overlay.mp4"

    extractor = PoseGraphExtractor(MODEL_PATH)
    data = extractor.process_video(VIDEO_PATH, OUTPUT_VIDEO)
    extractor.save_json(data, OUTPUT_JSON)
    print("Proceso completado.")