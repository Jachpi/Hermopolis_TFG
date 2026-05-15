#!/bin/bash
# Uso: sudo bash train.sh
set -e

if ! docker image inspect hermopolis_tfg:latest > /dev/null 2>&1; then
    echo "Imagen hermopolis_tfg no encontrada."
    echo "¿Con qué modo construirla?"
    echo "  1) gpu  (requiere NVIDIA + drivers CUDA)"
    echo "  2) cpu"
    read -p "Opción [1/2, por defecto 1]: " mode_choice
    case "$mode_choice" in
        2) BUILD_MODE=cpu ;;
        *) BUILD_MODE=gpu ;;
    esac
    echo "Construyendo con MODE=${BUILD_MODE}..."
    docker build --build-arg MODE="$BUILD_MODE" -t hermopolis_tfg "$(pwd)"
fi

echo "Selecciona el dataset de entrenamiento:"
echo "  1) default  (Poses clasificadas de forma normal) (taichi_dataset_default.pkl)"
echo "  2) loso (Leave One Subject Out) (taichi_dataset_loso.pkl)"
read -p "Opción [1/2, por defecto 1]: " choice

case "$choice" in
    2) TAICHI_DATASET=loso ;;
    *) TAICHI_DATASET=default ;;
esac

echo "Usando: taichi_dataset_${TAICHI_DATASET}.pkl"

WORK_DIRS="$(pwd)/mmaction2/work_dirs"
mkdir -p "$WORK_DIRS"

echo ""
echo "Entrando al contenedor. Lanza los entrenamientos con:"
echo "  python tools/train.py configs/skeleton/stgcn/stgcn_taichi_jm.py"
echo "  python tools/train.py configs/skeleton/stgcn/stgcn_taichi_bm.py"
echo "  python tools/train.py configs/skeleton/stgcn/stgcn_taichi_b.py"
echo "  python tools/train.py configs/skeleton/stgcn/stgcn_taichi.py"
echo ""

docker run -it --gpus all \
  -e TAICHI_DATASET="$TAICHI_DATASET" \
  -v "$WORK_DIRS":/workspace/Hermopolis_TFG/mmaction2/work_dirs \
  hermopolis_tfg \
  bash
