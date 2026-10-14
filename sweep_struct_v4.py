"""C-only sweeps of two structured knobs, paired against the baseline on the SAME shots (no torch needed).

  --damp   : message damping g (etar = g*new + (1-g)*old), 1.0 = off
  --blocks : separate alpha for (A rows, B rows, tie rows) as 'aA,aB,aT'.
             Default grid: change one block at a time, the others stay at --base_alpha.
Run from the repo root:
    python sweep_struct_v4.py --d 6 --p 0.004 --shots 262144 --threads 8
    python sweep_struct_v4.py --d 6 --p 0.004 --shots 262144 --threads 8 --damp 0.9 0.8 --blocks 0.9375,0.96875,1.0
"""
import argparse
import ctypes
import time
import numpy as np
from scipy.stats import binomtest

from nms_problem_v4 import build_problem, Sampler
from my_decoders import hbplib_wrapper_v4 as W
from diagnose_v4 import decode

lib = W.lib
lib.hbp_set_alpha_table_g.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.hbp_set_alpha_table_g.restype = None
lib.hbp_set_damping.argtypes = [ctypes.c_double]
lib.hbp_set_damping.restype = None


def reset():
    lib.hbp_set_alpha_table_g(None, 0, 1)
    lib.hbp_set_damping(1.0)


def set_blocks(t):
    arr = np.ascontiguousarray(t, dtype=np.float64).reshape(1, 3)
    lib.hbp_set_alpha_table_g(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--base_alpha", type=float, default=0.96875)
    ap.add_argument("--damp", type=float, nargs="*", default=[0.9, 0.8, 0.7, 0.5])
    ap.add_argument("--blocks", nargs="*", default=None, help="list of 'aA,aB,aT'")
    ap.add_argument("--shots", type=int, default=262144)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--mitr", type=int, default=400)
    ap.add_argument("--cs", type=int, default=2)
    ap.add_argument("--rs", type=int, default=1)
    ap.add_argument("--et", type=int, default=1)
    ap.add_argument("--prt", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()

    b = a.base_alpha
    if a.blocks is None:
        grid = []
        for v in (0.9, 0.9375, 1.0):
            grid += [(v, b, b), (b, v, b), (b, b, v)]
        blocks = [tuple(x) for x in grid]
    else:
        blocks = [tuple(float(y) for y in x.split(",")) for x in a.blocks]
    configs = [("baseline", None, None)]
    configs += [(f"damp={g}", None, g) for g in a.damp]
    configs += [(f"blocks(A,B,tie)=({x[0]:g},{x[1]:g},{x[2]:g})", x, None) for x in blocks]

    prob = build_problem(a.d, a.p, a.prt)
    smp = Sampler(prob, seed=a.seed)
    fails = {c[0]: [] for c in configs}
    iters = {c[0]: [] for c in configs}
    done, t0 = 0, time.time()
    while done < a.shots:
        k = min(a.chunk, a.shots - done)
        syn, _, obs = smp.sample(k, need_target=False)
        for name, blk, g in configs:
            reset()
            if blk is not None:
                set_blocks(blk)
            if g is not None:
                lib.hbp_set_damping(g)
            f, it = decode(prob, syn, obs, b, a.mitr, a.rs, a.cs, a.et, a.threads)
            fails[name].append(f); iters[name].append(it)
        reset()
        done += k
        print(f"  {done}/{a.shots} shots ({time.time() - t0:.0f}s)", flush=True)
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    iters = {k: np.concatenate(v) for k, v in iters.items()}
    base = fails["baseline"]
    n = a.shots
    print("=" * 100)
    print(f"d={a.d} p={a.p} shots={n} mitr={a.mitr} base alpha={b}   (paired vs baseline on identical shots)")
    print(f"{'config':44} {'fails':>6} {'LER':>9} {'avg_it':>7} {'only_base':>9} {'only_cfg':>8} {'net':>5} {'p-value':>9}")
    for name, _, _ in configs:
        f = fails[name]
        ob, oc = int((base & ~f).sum()), int((~base & f).sum())
        pv = binomtest(oc, ob + oc, 0.5).pvalue if ob + oc else 1.0
        print(f"{name:44} {int(f.sum()):6d} {f.mean():9.5f} {iters[name].mean():7.2f} {ob:9d} {oc:8d} {ob - oc:+5d} {pv:9.3g}")
    print("net > 0 = fewer failures than baseline.  With many configs, expect a few p<0.05 by chance alone "
          "(Bonferroni: need p < 0.05/%d); confirm any winner on a new --seed." % (len(configs) - 1))
    print("=" * 100)


if __name__ == "__main__":
    main()
