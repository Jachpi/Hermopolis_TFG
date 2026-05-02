import cv2
import json
import numpy as np
from mmpose.apis import MMPoseInferencer


class MMPoseExtractor:

    def __init__(self, smooth_window=9):
        self.inferencer = MMPoseInferencer(pose3d='human3d')
        '''
        Inicializa el inferencer de MMPose con el modelo MotionBERT entrenado en Human3.6M.
        Este modelo estima poses 3D directamente desde vídeo RGB, devolviendo
        coordenadas en metros relativas al nodo raíz (pelvis). El orden de los 17 nodos
        está definido en el repositorio oficial de MMPose:
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/body3d/body3d_h36m_dataset.py

        Human3.6M keypoint indexes:
        0: root(pelvis), 1: right_hip, 2: right_knee, 3: right_foot,
        4: left_hip, 5: left_knee, 6: left_foot, 7: spine,
        8: thorax, 9: neck_base, 10: head,
        11: left_shoulder, 12: left_elbow, 13: left_wrist,
        14: right_shoulder, 15: right_elbow, 16: right_wrist

        El sistema de coordenadas es mano derecha con Y apuntando hacia arriba,
        por lo que no es necesario invertir ningún eje como se hacía con MediaPipe y Kinect.
        '''

        self.graph_sequence = []

        self.smooth_window = smooth_window
        '''
        Tamaño de la ventana del filtro para el suavizado de movimientos entre frames.
        Mayor smooth implica menos movimientos bruscos.
        '''

        '''
        Mapeo de nodos Human3.6M a nodos del grafo taichi (Kinect)
        Los nodos de pies y manos no existen en Human3.6M, así que se aproximan
        usando los nodos más cercanos disponibles.
        '''

        self.TAICHI_TO_H36M = {
            0:  0,   # SPINEBASE     <- root(pelvis)
            3:  10,  # HEAD          <- head
            4:  11,  # SHOULDERLEFT  <- left_shoulder
            5:  12,  # ELBOWLEFT     <- left_elbow
            6:  13,  # WRISTLEFT     <- left_wrist
            8:  14,  # SHOULDERRIGHT <- right_shoulder
            9:  15,  # ELBOWRIGHT    <- right_elbow
            10: 16,  # WRISTRIGHT    <- right_wrist
            12: 4,   # HIPLEFT       <- left_hip
            13: 5,   # KNEELEFT      <- left_knee
            14: 6,   # ANKLELEFT     <- left_foot
            15: 6,   # FOOTLEFT      <- left_foot (aproximacion)
            16: 1,   # HIPRIGHT      <- right_hip
            17: 2,   # KNEERIGHT     <- right_knee
            18: 3,   # ANKLERIGHT    <- right_foot
            19: 3,   # FOOTRIGHT     <- right_foot (aproximacion)
            21: 13,  # HANDTIPLEFT   <- left_wrist (aproximacion)
            22: 13,  # THUMBLEFT     <- left_wrist (aproximacion)
            23: 16,  # HANDTIPRIGHT  <- right_wrist (aproximacion)
            24: 16,  # THUMBRIGHT    <- right_wrist (aproximacion)
        }
 
    def build_nodes(self, kpts):
        
        '''
        Construye el diccionario de nodos taichi a partir de los keypoints de Human3.6M.
        kpts: lista de 17 elementos, cada uno [x, y, z] en metros.
        '''
        nodes = {}
 
        for taichi_idx, h36m_idx in self.TAICHI_TO_H36M.items():
            kpt = kpts[h36m_idx]
            nodes[taichi_idx] = {
                "x": float(kpt[0]),
                "y": float(kpt[1]),
                "z": float(kpt[2]),
                "visibility": 1.0
            }
 
        return nodes


    def smooth(self, data):
        '''
        Aplica un filtro de media movil sobre cada nodo a lo largo del tiempo.

        Para cada nodo y cada eje (x, y, z), se calcula la media de los
        smooth_window frames centrados en el frame actual. En los extremos
        se usa padding de borde, repitiendo el primer o ultimo valor disponible
        para no perder frames.
        '''
        if len(data) == 0 or self.smooth_window <= 1:
            return data
 
        w = self.smooth_window
        node_ids = list(self.TAICHI_TO_H36M.keys())

        # Construir arrays de coordenadas por nodo con shape (num_frames, 3)
        coords = {}
        for nid in node_ids:
            coords[nid] = np.zeros((len(data), 3))

        for t, frame in enumerate(data):
            for nid in node_ids:
                n = frame['nodes'].get(nid)
                if n:
                    coords[nid][t] = [n['x'], n['y'], n['z']]

        # Aplicar media movil: w//2 frames a la izquierda y w//2 a la derecha
        smoothed = {}
        for nid in node_ids:
            smoothed[nid] = np.zeros_like(coords[nid])
            for t in range(len(data)):
                left  = max(0, t - w//2)
                right = min(len(data), t + w//2 + 1)
                smoothed[nid][t] = coords[nid][left:right].mean(axis=0)
 
        # Reconstruir la secuencia en formato de diccionario
        smoothed_data = []
        for t, frame in enumerate(data):
            new_frame = {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "nodes": {}
            }
            for nid in node_ids:
                new_frame["nodes"][nid] = {
                    "x": float(smoothed[nid][t][0]),
                    "y": float(smoothed[nid][t][1]),
                    "z": float(smoothed[nid][t][2]),
                    "visibility": 1.0
                }
            smoothed_data.append(new_frame)
 
        return smoothed_data


    def save_json(self, data, output_path):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Grafo guardado en: {output_path}")

    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frame_idx = 0
        last_valid_kpts = None
        result_generator = self.inferencer(video_path, show=False, bbox_thr=0.1)

        for result in result_generator:
            predictions = result.get('predictions', [])

            if predictions and len(predictions[0]) > 0:
                kpts = predictions[0][0]['keypoints']
                last_valid_kpts = kpts
            elif last_valid_kpts is not None:
                kpts = last_valid_kpts
            else:
                frame_idx += 1
                continue

            self.graph_sequence.append({
                "frame_index": frame_idx,
                "timestamp_ms": int(frame_idx * 1000 / fps),
                "nodes": self.build_nodes(kpts)
            })

            frame_idx += 1

        return self.smooth(self.graph_sequence)


if __name__ == "__main__":
    VIDEO_PATH = "Hermopolis_TFG/videos/clip.mp4"
    OUTPUT_JSON = "mmpose_nodes.json"

    extractor = MMPoseExtractor(smooth_window=5)
    data = extractor.process_video(VIDEO_PATH)
    extractor.save_json(data, OUTPUT_JSON)
    print(f"Proceso completado. {len(data)} frames procesados.")