"""Read BabyView pose pickles and the COCO-WholeBody 133-keypoint layout.
Pickles store torch tensors on CUDA; we remap storages to CPU on load so no GPU is needed."""
import io, pickle
import numpy as np

POSE_ROOT = "/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old"
FRAME_ROOT = "/ccn2/dataset/babyview/outputs_20250312/sampled_frames"

# COCO-WholeBody, 0-based, 133 keypoints
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP = 11, 12
FACE = list(range(23, 91))          # 68 dense face landmarks
LHAND = list(range(91, 112))        # 21
RHAND = list(range(112, 133))       # 21
BODY_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pose(path):
    """Return dict: bboxes [N,4] xyxy, confs [N], and per-person keypoints [133,2] + scores [133]."""
    with open(path, "rb") as f:
        obj = _CPUUnpickler(f).load()
    det = obj.get("person_detection_dict", {}) or {}
    pdd = obj.get("pose_dict", {}) or {}

    def arr(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
        except Exception:
            pass
        return np.asarray(x, dtype=np.float32)

    bboxes = arr(det.get("person_bboxes", [])).reshape(-1, 4) if len(det.get("person_bboxes", [])) else np.zeros((0, 4), np.float32)
    confs = arr(det.get("person_confs", [])).reshape(-1) if len(det.get("person_confs", [])) else np.zeros((0,), np.float32)
    persons = []
    for pid, pv in pdd.items():
        kp = arr(pv["keypoints"]).reshape(-1, 2)
        sc = arr(pv["keypoint_scores"]).reshape(-1)
        persons.append({"id": pid, "kp": kp, "score": sc})
    return {"bboxes": bboxes, "confs": confs, "persons": persons}


def frame_path(gcp_processed, sec):
    return f"{FRAME_ROOT}/{gcp_processed}/{sec}.jpg"


def pose_path(gcp_processed, sec):
    return f"{POSE_ROOT}/{gcp_processed}/{sec}.pkl"


def draw_overlay(frame_path_, pose, out_path, kp_thresh=0.3):
    """Draw skeleton + keypoints over the frame (for human sanity check). PIL only."""
    from PIL import Image, ImageDraw
    try:
        im = Image.open(frame_path_).convert("RGB")
    except Exception:
        im = Image.new("RGB", (512, 512), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for b in pose["bboxes"]:
        d.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline=(0, 255, 0), width=2)
    for P in pose["persons"]:
        kp, sc = P["kp"], P["score"]
        for a, b in BODY_EDGES:
            if a < len(sc) and b < len(sc) and sc[a] > kp_thresh and sc[b] > kp_thresh:
                d.line([tuple(kp[a]), tuple(kp[b])], fill=(255, 80, 80), width=3)
        groups = [(range(0, 17), (255, 255, 0), 3), (FACE, (0, 200, 255), 1),
                  (LHAND, (255, 0, 255), 2), (RHAND, (0, 255, 128), 2)]
        for idxs, col, r in groups:
            for i in idxs:
                if i < len(sc) and sc[i] > kp_thresh:
                    x, y = float(kp[i, 0]), float(kp[i, 1])
                    d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    im.save(out_path, quality=85)
