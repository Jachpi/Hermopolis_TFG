import subprocess
import argparse
import os

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")

parser = argparse.ArgumentParser(description="Descarga y recorta un clip de YouTube usando yt-dlp.")
parser.add_argument("--ytID", required=True, help="ID del vídeo de YouTube (p.ej. OVJ4PWYJxnY)")
parser.add_argument("--output", required=True, help="Nombre del archivo de salida (p.ej. clip.mp4)")
parser.add_argument("--start", type=int, default=0, help="Frame de inicio (por defecto 0)")
parser.add_argument("--end", type=int, required=True, help="Frame de fin")
parser.add_argument("--fps", type=float, default=30.0, help="FPS del vídeo (por defecto 30)")
args = parser.parse_args()

video_id = args.ytID
output_file = os.path.join(SAMPLES_DIR, args.output)
start_s = args.start / args.fps
duration_s = (args.end - args.start) / args.fps

url = f"https://www.youtube.com/watch?v={video_id}"
print(f"Descargando y recortando {url}")

subprocess.run([
    "yt-dlp",
    "-f", "mp4",
    "--download-sections", f"*{start_s}-{start_s + duration_s}",
    "-o", output_file,
    url
], check=True)

print(f"Clip guardado en {output_file}")
