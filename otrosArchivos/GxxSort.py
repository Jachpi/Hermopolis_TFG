import os
import re
import shutil

DIR = r"../../Segmented_Kinect/Segmented_Kinect"

regex = re.compile(r"P..T..C..(G..)D..S..",re.IGNORECASE)

def classify_kinect_txt(DIR="./"):
    if os.path.exists(DIR) and os.path.isdir(DIR):
        abs_path = os.path.abspath(DIR)
        files = os.listdir(abs_path)
        for f in files:
            m = regex.match(f)
            if f.lower().endswith(".txt") and m:
                new_folder = os.path.join(abs_path,m.group(1))
                os.makedirs(new_folder, exist_ok=True)
                src_p = os.path.join(abs_path,f)
                dst_p = os.path.join(new_folder,f)
                if not os.path.exists(dst_p):
                    shutil.move(src=src_p,dst=dst_p)
    print("Clasificación completada")

if __name__ == "__main__":
    classify_kinect_txt(DIR)
