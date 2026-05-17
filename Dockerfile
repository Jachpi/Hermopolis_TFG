ARG MODE=gpu
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# git, ninja-build y curl solo son necesarios durante el build.
# Se instalan, se usan y se purgan en la misma capa para no inflar la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
        libsndfile1 libturbojpeg libgles2 libegl1 \
        git ninja-build curl \
    && pip install --no-cache-dir "openmim>=0.3.9" \
    && mim install "mmcv>=2.0.0" \
    && pip install --no-cache-dir "mmengine>=0.7.2" "mmdet>=3.0.0" \
    && mim install mmpose \
    # ── yt-dlp: binario standalone, independiente de Python 3.9 ──
    && curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
       -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp \
    # ── limpieza: purgamos todo lo que solo era necesario en build ──
    && apt-get purge -y git ninja-build curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Hermopolis_TFG

COPY mmaction2 /workspace/Hermopolis_TFG/mmaction2
RUN pip install --no-cache-dir -e ./mmaction2

RUN pip install --no-cache-dir \
    "mediapipe>=0.10.0" \
    "opencv-contrib-python==4.8.1.78" \
    "numpy==1.26.4" \
    "torchvision==0.16.0" \
    "scipy" \
    "einops" \
    "decord>=0.4.1" \
    "seaborn" \
    "importlib_metadata"

COPY pipelineStruct  /workspace/Hermopolis_TFG/pipelineStruct
COPY samples         /workspace/Hermopolis_TFG/samples

WORKDIR /workspace/Hermopolis_TFG/mmaction2