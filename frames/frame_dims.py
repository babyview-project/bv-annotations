# per-video frame dims (w,h) from the first jpg of each video dir; args: <frames_root> <out_tsv>
import os, sys
from PIL import Image
root, out = sys.argv[1], sys.argv[2]
done = set()
if os.path.exists(out):
    done = {l.split('\t')[0] for l in open(out)}
with open(out, 'a') as f:
    for d in sorted(os.listdir(root)):
        if d in done: continue
        p = os.path.join(root, d)
        try:
            j = next(x for x in sorted(os.listdir(p)) if x.endswith('.jpg'))
            w, h = Image.open(os.path.join(p, j)).size
            n = sum(1 for x in os.listdir(p) if x.endswith('.jpg'))
        except StopIteration:
            w = h = n = 0
        f.write(f'{d}\t{w}\t{h}\t{n}\n')
print('DIMS_DONE')
