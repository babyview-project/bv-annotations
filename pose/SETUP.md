# pose env setup (ccn2)

mmpose is genuinely finicky: the hard constraint is that **`mmcv`'s compiled ops must match
the installed `torch`**. Getting this wrong gives an import-time
`undefined symbol: …` from `mmcv/_ext…so` (an ABI mismatch — exactly what khaiaw's original
env now shows, because its torch was upgraded past what its mmcv was built for).

The combination below is known-good on ccn2 (A40, CUDA 12.x driver): **torch 2.1.0 + cu121 +
mmcv 2.1.0** (installed from openmmlab's *prebuilt wheel index*, not built from source).

```bash
# node-local NVMe; use a roomy TMPDIR because /tmp is small and /data2 can be near-full
mkdir -p /data2/mcfrank/tmp
PY=/data2/mcfrank/ladder/condaenv/bin/python          # any 3.11 python to seed the venv
$PY -m venv /data2/mcfrank/mmpose_env
ENV=/data2/mcfrank/mmpose_env/bin
PIP="TMPDIR=/data2/mcfrank/tmp $ENV/pip install --no-cache-dir --cache-dir /data2/mcfrank/tmp/pipcache"

# 1. torch 2.1.0 (cu121 wheels run fine on the 12.x driver)
eval $PIP torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu   # ← use /cu121 not /cpu
#   NOTE: real command uses  --index-url https://download.pytorch.org/whl/cu121

# 2. mmcv 2.1.0 from the PREBUILT wheel index for exactly cu121/torch2.1.0 (do NOT let pip/mim
#    build it from source — that fails). This drags numpy to 2.x; we fix that in step 4.
eval $PIP "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# 3. mmdet + mmpose. --no-build-isolation so the build sees the env's torch/mmcv/numpy.
#    Needs `wheel` present first (chumpy, an ancient mmpose dep, builds a wheel).
eval $PIP wheel setuptools
eval $PIP --no-build-isolation "mmdet==3.3.0" "mmpose==1.3.2"

# 4. pin numpy<2 LAST (torch 2.1 + mm* were compiled against numpy 1.x; numpy 2 breaks them)
eval $PIP "numpy==1.26.4"

# 5. detector + parallelism
eval $PIP ultralytics ray
```

Verify (the real test is that `mmcv._ext` loads — plain `import mmcv` isn't enough):

```bash
CUDA_VISIBLE_DEVICES=0 /data2/mcfrank/mmpose_env/bin/python -c "
import torch, numpy, mmcv, mmdet, mmpose
from mmcv.ops import nms                       # forces the _ext ABI load
print(torch.__version__, numpy.__version__, mmcv.__version__, mmpose.__version__, torch.cuda.is_available())"
# expect: 2.1.0+cu121 1.26.4 2.1.0 1.3.2 True
```

`requirements.txt` is the full `pip freeze` of the working env, but reproduce via the ordered
steps above — a flat `pip install -r` will re-trigger the mmcv source-build and the numpy race.

## Models

`run_model.py` loads (relative to CWD): `yolo12x.pt` and `downloads/rtmw-x_…256x192.py` +
`…256x192-13a2546d_20231208.pth`. On ccn2 these live in khaiaw's repo; the working copy at
`/data2/mcfrank/babyview-pose/` symlinks them:

```bash
ln -s /ccn2/u/khaiaw/Code/babyview-pose/pose-detection/downloads  downloads
ln -s /ccn2/u/khaiaw/Code/babyview-pose/pose-detection/yolo12x.pt yolo12x.pt
```

To fetch fresh: YOLO12x via `ultralytics` (`YOLO('yolo12x.pt')` auto-downloads); RTMW-x config
+ checkpoint from the mmpose [rtmw cocktail14](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md)
page (use the non-distilled `rtmw-x`, not `rtmw-dw-*` — see `qa/README.md`).
