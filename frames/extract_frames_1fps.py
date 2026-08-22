#!/usr/bin/env python3
"""1-fps frame extraction for a BabyView release, reproducing khaiaw's 2025.2 recipe exactly
(babyview-dataset-user/extract_frames_from_videos.py, Sept 2025): fps=1:round=near, short side -> 512,
JPEG qscale 1, yuvj444p, vsync vfr, %05d.jpg (index = second). Adds: explicit video list, resumability
via per-video marker files, a manifest (video, w, h, n_frames, secs), -y/-nostdin, thread cap, nice.

usage: extract_frames_1fps.py --list videos.txt --out_root <.../extracted_frames_1fps> \
         --marker_dir <node-local dir> --manifest <tsv> [--num_processes 16] [--threads 4]
"""
import argparse, os, subprocess, time, sys, glob
from multiprocessing import Pool, Value

GPU_ID = None           # set per worker when --hwaccel_gpus is given
_counter = None

def _init_worker(counter, gpus):
    global GPU_ID
    with counter.get_lock():
        i = counter.value; counter.value += 1
    GPU_ID = gpus[i % len(gpus)] if gpus else None

SCALE = "scale='if(gte(iw,ih),-1,512):if(gte(iw,ih),512,-1)'"

def extract(video_path):
    vid = os.path.basename(video_path).rsplit('.', 1)[0]
    out_dir = os.path.join(ARGS.out_root, vid)
    marker = os.path.join(ARGS.marker_dir, vid)
    if os.path.exists(marker):
        return (vid, 'skip', 0, 0, 0, 0.0)
    # another instance (forward/reverse over the same list) is on it: dir exists, no marker, recently touched
    if os.path.isdir(out_dir) and time.time() - os.path.getmtime(out_dir) < ARGS.inprogress_min * 60:
        return (vid, 'skip_inprogress', 0, 0, 0, 0.0)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    hw = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'] if GPU_ID is not None else []
    vf = f'fps=1:round=near,hwdownload,format=nv12,{SCALE}' if GPU_ID is not None else f'fps=1:round=near,{SCALE}'
    cmd = ['nice', '-n', str(ARGS.nice), 'ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error'] + hw + [
           '-threads', str(ARGS.threads), '-i', video_path,
           '-vf', vf, '-qscale:v', '1', '-pix_fmt', 'yuvj444p',
           '-video_track_timescale', '1000', '-frame_pts', '1', '-vsync', 'vfr',
           os.path.join(out_dir, '%05d.jpg')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_ID)) if GPU_ID is not None else None
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    secs = time.time() - t0
    frames = sorted(glob.glob(os.path.join(out_dir, '*.jpg')))
    w = h = 0
    if frames:
        try:
            from PIL import Image
            w, h = Image.open(frames[0]).size
        except Exception:
            pass
    status = 'ok' if (r.returncode == 0 and frames) else f'fail_rc{r.returncode}'
    if status == 'ok':
        with open(marker, 'w') as f:
            f.write(f'{len(frames)}\t{w}\t{h}\t{secs:.1f}\n')
    else:
        sys.stderr.write(f'FAIL {vid} rc={r.returncode} {r.stderr[-500:]}\n')
    return (vid, status, len(frames), w, h, secs)

def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument('--list', required=True); p.add_argument('--out_root', required=True)
    p.add_argument('--marker_dir', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--num_processes', type=int, default=16); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--nice', type=int, default=10)
    p.add_argument('--reverse', action='store_true', help='process the list back-to-front (run a 2nd instance this way)')
    p.add_argument('--inprogress_min', type=float, default=30, help='treat a marker-less dir younger than this as in progress')
    p.add_argument('--shuffle', type=int, default=None, help='seed: process the list in a seeded random order (3rd instance)')
    p.add_argument('--hwaccel_gpus', default='', help='comma-separated GPU ids: decode on NVDEC (variant A, byte-identical), round-robin per worker')
    ARGS = p.parse_args()
    os.makedirs(ARGS.out_root, exist_ok=True); os.makedirs(ARGS.marker_dir, exist_ok=True)
    videos = [l.strip() for l in open(ARGS.list) if l.strip()]
    if ARGS.reverse: videos = videos[::-1]
    if ARGS.shuffle is not None:
        import random; random.Random(ARGS.shuffle).shuffle(videos)
    gpus = [int(g) for g in ARGS.hwaccel_gpus.split(',') if g != '']
    print(f'{len(videos)} videos, {ARGS.num_processes} procs x {ARGS.threads} threads', flush=True)
    done = 0; t0 = time.time()
    counter = Value('i', 0)
    with open(ARGS.manifest, 'a') as mf, Pool(ARGS.num_processes, initializer=_init_worker, initargs=(counter, gpus)) as pool:
        for vid, status, n, w, h, secs in pool.imap_unordered(extract, videos, chunksize=1):
            done += 1
            if not status.startswith('skip'):
                mf.write(f'{vid}\t{status}\t{n}\t{w}\t{h}\t{secs:.1f}\n'); mf.flush()
            if done % 25 == 0 or status.startswith('fail'):
                print(f'[{done}/{len(videos)}] {vid} {status} n={n} {w}x{h} {secs:.0f}s  elapsed {(time.time()-t0)/60:.1f} min', flush=True)
    print('ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
