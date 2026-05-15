ARG MODE=gpu

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base-gpu
FROM pytorch/pytorch:2.1.0 AS base-cpu

FROM base-${MODE}

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# git y ninja-build solo son necesarios durante la instalación del stack OpenMMLab.
# Se instalan, se usan y se purgan en la misma capa para no inflar la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
    libsndfile1 libturbojpeg git ninja-build \
    && pip install --no-cache-dir "openmim>=0.3.9" \
    && mim install "mmcv>=2.0.0" \
    && pip install --no-cache-dir "mmengine>=0.7.2" "mmdet>=3.0.0" \
    && mim install mmpose \
    && apt-get purge -y git ninja-build \
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