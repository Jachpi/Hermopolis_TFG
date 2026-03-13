import os
import json

BASE_DIR = "../Segmented_Kinect/Segmented_Kinect/"
OUTPUT_PATH = "frame_distribution.json"

frame_distribution = {}
min_frames = (None, float("inf"))   # (nombre_archivo, frames)
max_frames = (None, float("-inf"))  # (nombre_archivo, frames)

# Para la media por carpeta
folder_stats = {}

for folder in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    folder_frames = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            frame_count = sum(1 for line in f if line.strip())

        print(f"Leyendo: {file_path} → {frame_count} frames")

        key = str(frame_count)
        frame_distribution[key] = frame_distribution.get(key, 0) + 1

        if frame_count < min_frames[1]:
            min_frames = (file_path, frame_count)
        if frame_count > max_frames[1]:
            max_frames = (file_path, frame_count)

        folder_frames.append(frame_count)

    if folder_frames:
        folder_stats[folder] = {
            "media": round(sum(folder_frames) / len(folder_frames), 2),
            "min": min(folder_frames),
            "max": max(folder_frames),
            "num_clips": len(folder_frames)
        }

# Ordenar por número de frames para que el json sea más legible
frame_distribution = dict(sorted(frame_distribution.items(), key=lambda x: int(x[0])))

with open(OUTPUT_PATH, "w") as f:
    json.dump(frame_distribution, f, indent=2)

print(f"\nDistribución guardada en {OUTPUT_PATH}")
print(f"Clip con menos frames: {min_frames[0]} → {min_frames[1]} frames")
print(f"Clip con más frames:   {max_frames[0]} → {max_frames[1]} frames")

print(f"\nMedia de frames por carpeta:")
for folder, stats in folder_stats.items():
    print(f"  {folder}: media={stats['media']} | min={stats['min']} | max={stats['max']} | clips={stats['num_clips']}")