import json
import matplotlib.pyplot as plt

'''
val_epochs = []
val_acc = []
train_epochs = []
train_loss = []

with open('../mmaction2/work_dirs/stgcn_taichi/20260330_191110/vis_data/scalars.json') as f:
    for line in f:
        d = json.loads(line)
        if 'acc/top1' in d:
            val_epochs.append(d['step'])
            val_acc.append(d['acc/top1'])
        if 'loss' in d and 'acc/top1' not in d:
            train_epochs.append(d['epoch'])
            train_loss.append(d['loss'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(val_epochs, val_acc)
ax1.set_title('Precisión en validación (acc/top1)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.grid(True)

ax2.plot(train_epochs, train_loss)
ax2.set_title('Loss de entrenamiento')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)

plt.tight_layout()
plt.savefig('curvas_j.png', dpi=150)
print('Guardado en curvas_j.png')
'''

'''val_epochs = []
val_acc = []
train_epochs = []
train_loss = []

with open('../mmaction2/work_dirs/stgcn_taichi_jm/20260424_182833/vis_data/scalars.json') as f:
    for line in f:
        d = json.loads(line)
        if 'acc/top1' in d:
            val_epochs.append(d['step'])
            val_acc.append(d['acc/top1'])
        if 'loss' in d and 'acc/top1' not in d:
            train_epochs.append(d['epoch'])
            train_loss.append(d['loss'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(val_epochs, val_acc)
ax1.set_title('Precisión en validación (acc/top1) - jm')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.grid(True)

ax2.plot(train_epochs, train_loss)
ax2.set_title('Loss de entrenamiento - jm')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)

plt.tight_layout()
plt.savefig('curvas_jm.png', dpi=150)
print('Guardado en curvas_jm.png')
'''

val_epochs = []
val_acc = []
train_epochs = []
train_loss = []

with open('../mmaction2/work_dirs/stgcn_taichi_bm/20260502_191025/vis_data/scalars.json') as f:
    for line in f:
        d = json.loads(line)
        if 'acc/top1' in d:
            val_epochs.append(d['step'])
            val_acc.append(d['acc/top1'])
        if 'loss' in d and 'acc/top1' not in d:
            train_epochs.append(d['epoch'])
            train_loss.append(d['loss'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(val_epochs, val_acc)
ax1.set_title('Precisión en validación (acc/top1) - bm')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.grid(True)

ax2.plot(train_epochs, train_loss)
ax2.set_title('Loss de entrenamiento - bm')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)

plt.tight_layout()
plt.savefig('curvas_bm.png', dpi=150)
print('Guardado en curvas_bm.png')