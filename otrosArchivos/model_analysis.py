import pickle
import numpy as np

GESTURE_NAMES = [
    'G01 - Beginning position',
    'G02 - Tree posture',
    'G03 - Open and close lotus flower',
    'G04 - Bring sky and earth together',
    'G05 - Canalize energy',
    'G06 - Drive the monkey away',
    'G07 - Move hands like clouds',
    'G08 - Part the wild horse mane',
    'G09 - Golden rooster',
    'G10 - Fair lady works shuttles',
    'G11 - Kick with heel',
    'G12 - Brush knee and twist step',
    'G13 - Grasp the bird tail'
]

with open('resultados.pkl', 'rb') as f:
    resultados = pickle.load(f)

# Matriz de confusión 13x13
confusion = np.zeros((13, 13), dtype=int)
for item in resultados:
    gt = item['gt_label'].item()
    pred = item['pred_label'].item()
    confusion[gt][pred] += 1

# Imprimir por clase
print(f"{'Gesto':<40} {'Correctas':>10} {'Total':>8} {'Precisión':>10}")
print("-" * 70)
for i in range(13):
    total = confusion[i].sum()
    correctas = confusion[i][i]
    prec = correctas / total if total > 0 else 0
    print(f"{GESTURE_NAMES[i]:<40} {correctas:>10} {total:>8} {prec:>10.1%}")

print("\nConfusiones más frecuentes:")
for i in range(13):
    for j in range(13):
        if i != j and confusion[i][j] > 0:
            print(f"  {GESTURE_NAMES[i]} → predicho como {GESTURE_NAMES[j]}: {confusion[i][j]} veces")