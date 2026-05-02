import os
import json

base = 'mmaction2/work_dirs/stgcn_taichi_jm'
for dirname in sorted(os.listdir(base)):
    dirpath = os.path.join(base, dirname, 'vis_data', 'scalars.json')
    if os.path.isfile(dirpath):
        with open(dirpath) as f:
            lines = f.readlines()
        val_entries = [json.loads(l) for l in lines if 'acc/top1' in l]
        if val_entries:
            last = val_entries[-1]
            print(f"{dirname}: acc/top1={last['acc/top1']:.4f}, entries={len(lines)}, step={last['step']}")
        else:
            print(f"{dirname}: sin entradas de validación, entries={len(lines)}")