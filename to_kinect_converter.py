import json
import numpy as np

OUTPUT_JSON = "pose_graph_translated.json"


class CustomKinectConverter:
    def __init__(self):
        self.kinectnodes = {
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
        self.nodetranslations = {
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
    def convertfromMediapipe(self,file):
        with open(file,"r") as d:
            data = json.load(d)
            translated_data = []
            for frame in data:
                new_frame = {"frame_index": frame["frame_index"], "timestamp_ms": frame["timestamp_ms"], "nodes": {}}
                for node in frame["nodes"]:
                    new_frame["nodes"][self.kinectnodes[self.nodetranslations[int(node)]]] = frame["nodes"][node]
                translated_data.append(new_frame)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(translated_data, f, indent=2)
            
if __name__ == "__main__":
    c = CustomKinectConverter()
    c.convertfromMediapipe("pose_graph_sequence.json")