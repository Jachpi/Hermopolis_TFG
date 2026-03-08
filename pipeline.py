import argparse
import normalizer


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--type", required=True)
args = parser.parse_args()


INPUT_PATH = args.input
OUTPUT_JSON_M = args.output
INPUT_FLAG = args.type # Sólo puede ser o --from mediapipe o bien --from kinect

if INPUT_FLAG == "mediapipe":
    from example_pose_estimation import PoseGraphExtractor
    import to_kinect_converter

    poseG = PoseGraphExtractor("pose_landmarker_heavy.task")
    initial_data = poseG.process_video(INPUT_PATH)
    kinectizised_data = to_kinect_converter.convert_from_mediapipe(initial_data)
    normalized_data = normalizer.normalize(kinectizised_data)
    normalizer.save_json(normalized_data,OUTPUT_JSON_M)
elif INPUT_FLAG == "kinect":
    import kinect_transformer

    json_kinect = kinect_transformer.kinect_transform(INPUT_PATH)
    normalized_data = normalizer.normalize(json_kinect)
    normalizer.save_json(normalized_data, OUTPUT_JSON_M)

