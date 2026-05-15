import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# Nombres cortos para los ejes de la matriz
GESTURE_SHORT = [
    'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07',
    'G08', 'G09', 'G10', 'G11', 'G12', 'G13'
]

with open('../otrosArchivos/resultados_bm.pkl', 'rb') as f:
    resultados = pickle.load(f)

# Construir matriz de confusión
confusion = np.zeros((13, 13), dtype=int)
for item in resultados:
    gt = item['gt_label'].item()
    pred = item['pred_label'].item()
    confusion[gt][pred] += 1

# Normalizar por fila para obtener porcentajes
confusion_norm = confusion.astype(float)
for i in range(13):
    total = confusion[i].sum()
    if total > 0:
        confusion_norm[i] = confusion[i] / total

# Figura
fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(confusion_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(13))
ax.set_yticks(np.arange(13))
ax.set_xticklabels(GESTURE_SHORT, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(GESTURE_SHORT, fontsize=9)

ax.set_xlabel('Predicción', fontsize=11)
ax.set_ylabel('Etiqueta real', fontsize=11)
ax.set_title('Matriz de confusión — feats=[bm]', fontsize=13, pad=12)

# Escribir valores en cada celda
for i in range(13):
    for j in range(13):
        count = confusion[i][j]
        if count > 0:
            color = 'white' if confusion_norm[i][j] > 0.5 else 'black'
            ax.text(j, i, str(count), ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('matriz_confusion_bm.png', dpi=150)
print('Guardado en matriz_confusion_bm.png')

# Imprimir resumen por clase
print(f"\n{'Gesto':<40} {'Correctas':>10} {'Total':>8} {'Precision':>10}")
print("-" * 72)
for i in range(13):
    total = confusion[i].sum()
    correctas = confusion[i][i]
    prec = correctas / total if total > 0 else 0
    print(f"{GESTURE_NAMES[i]:<40} {correctas:>10} {total:>8} {prec:>10.1%}")

print("\nConfusiones:")
for i in range(13):
    for j in range(13):
        if i != j and confusion[i][j] > 0:
            print(f"  {GESTURE_NAMES[i]} → {GESTURE_NAMES[j]}: {confusion[i][j]} veces")