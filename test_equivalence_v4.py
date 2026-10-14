"""Check that GariNMSTorch reproduces the C decoder bit-for-bit (up to float rounding).
Run from the repo root (needs my_decoders/hbplib_v4.so built from the patched hbp_decoder_v4.c):
    python test_equivalence_v4.py                # random synthetic 'big matrix'
    python test_equivalence_v4.py --npz my.npz   # your real one: Hrows, llr, hshape0, dxshape0 (+ optional syn)
"""
import argparse
import numpy as np
import torch
try:
    from my_decoders import hbplib_wrapper_v4 as W
except ImportError:
    import hbplib_wrapper_v4 as W
from neural_nms_v4 import GariNMSTorch, IterBeta


def synthetic(seed=0, N=240, h=90, dx=45, nb=40, p=0.03):
    rng = np.random.default_rng(seed)
    H = np.zeros((h + nb, N), dtype=np.uint8)
    for r in range(h):
        H[r, rng.choice(N, rng.integers(3, 8), replace=False)] = 1
    for r in range(nb):
        H[h + r, rng.choice(N, 3, replace=False)] = 1
    D = int(H.sum(1).max())
    Hrows = np.zeros((H.shape[0], D), dtype=np.int32)
    for r in range(H.shape[0]):
        c = np.flatnonzero(H[r]) + 1
        Hrows[r, : len(c)] = c
    pr = rng.uniform(0.005, 0.05, N)
    llr = np.log((1 - pr) / pr)
    return H, Hrows, llr, h, dx, p, rng


def make_syn(H, h, p, rng, shots):
    e = (rng.random((shots, H.shape[1])) < p).astype(np.uint8)
    s = (e @ H.T) % 2
    s[:, h:] = 0                                   # extra rows have syndrome 0 (+1)
    return (1 - 2 * s).astype(np.int8)


def run_c(Hrows, llr, syn, Nloop, alpha, h, dx, sched, es, ro=0):
    Hcols = np.zeros((int(Hrows.max()), 1), dtype=np.int32)  # unused by the C decoder
    out = [W.ldpc_dec_msaa_quantum_serial_big_matrix_c(llr, Nloop, Hrows, Hcols, alpha, s, ro,
                                                       (h,), (dx,), sched, es) for s in syn]
    return (np.stack([o[0] for o in out]), np.array([o[1] for o in out]), np.stack([o[2] for o in out]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    a = ap.parse_args()
    if a.npz:
        d = np.load(a.npz)
        Hrows, llr, h, dx = d["Hrows"], d["llr"], int(d["hshape0"]), int(d["dxshape0"])
        rng = np.random.default_rng(1)
        syn = d["syn"] if "syn" in d else None
        assert syn is not None, "provide syn (shots, R) with +-1 entries in the npz"
    else:
        H, Hrows, llr, h, dx, p, rng = synthetic()
        syn = make_syn(H, h, p, rng, 48)
    N = len(llr)
    Nloop = 30
    ok = True
    cases = [(ro, sc, es, m) for ro in (0, 1, 7) for sc in (0, 1, 2, 3) for es in (False, True) for m in ("scalar", "table")]
    for ro, sched, es, mode in cases:
        if True:
            if True:
                dec = GariNMSTorch(Hrows, h, dx, N=N, custom_schedule=sched)
                if mode == "scalar":
                    IterBeta.clear_c(W.lib)
                    alpha, betas = 0.9375, 0.9375
                else:
                    bm = IterBeta(Nloop + 1, init=0.9)
                    with torch.no_grad():
                        bm.raw.add_(torch.randn_like(bm.raw) * 0.4)
                    bm.push_to_c(W.lib)
                    alpha, betas = 0.5, bm().detach()           # alpha must be ignored when the table is set
                hc, ic, lc = run_c(Hrows, llr, syn, Nloop, alpha, h, dx, sched, es, ro)
                with torch.no_grad():
                    r = dec(llr, syn, betas, Nloop, early_stopping=es, random_order=ro)
                covered = np.zeros(N, dtype=bool)               # C leaves Hdec uninitialised on columns no check touches
                covered[np.unique(Hrows[Hrows > 0]) - 1] = True
                same_hard = (r["hard"].numpy()[:, covered] == hc[:, covered]).all()
                same_it = (r["iters"].numpy() == ic).all()
                err = np.abs(r["lam"].numpy() - lc).max()
                good = same_hard and same_it and err < 1e-6
                ok &= good
                print(f"ro={ro} sched={sched} early_stop={es!s:5} {mode:6} levels={dec.num_levels:3d} "
                      f"hard_equal={same_hard} iters_equal={same_it} max|dlam|={err:.2e}  "
                      f"conv={int(r['converged'].sum())}/{len(syn)}  {'OK' if good else 'FAIL'}")
    # ---- block-wise betas (A rows / B rows / tie rows) and message damping
    covered = np.zeros(N, dtype=bool)
    covered[np.unique(Hrows[Hrows > 0]) - 1] = True
    import ctypes
    for ro, sched, es, groups, damp in [(0, 2, True, 3, 1.0), (7, 2, True, 3, 1.0), (7, 2, True, 3, 0.8),
                                        (1, 0, False, 1, 0.7), (7, 3, True, 3, 0.9)]:
        dec = GariNMSTorch(Hrows, h, dx, N=N, custom_schedule=sched)
        bm = IterBeta(Nloop + 1, init=0.9, num_groups=groups)
        with torch.no_grad():
            bm.raw.add_(torch.randn_like(bm.raw) * 0.4)
        IterBeta.clear_c(W.lib)
        bm.push_to_c(W.lib)
        W.lib.hbp_set_damping.argtypes = [ctypes.c_double]
        W.lib.hbp_set_damping(damp)
        hc, ic, lc = run_c(Hrows, llr, syn, Nloop, 0.5, h, dx, sched, es, ro)
        with torch.no_grad():
            r = dec(llr, syn, bm().detach(), Nloop, early_stopping=es, random_order=ro, damping=damp)
        same_hard = (r["hard"].numpy()[:, covered] == hc[:, covered]).all()
        same_it = (r["iters"].numpy() == ic).all()
        err = np.abs(r["lam"].numpy() - lc).max()
        good = same_hard and same_it and err < 1e-6
        ok &= good
        print(f"ro={ro} sched={sched} early_stop={es!s:5} groups={groups} damping={damp}  hard_equal={same_hard} "
              f"iters_equal={same_it} max|dlam|={err:.2e}  {'OK' if good else 'FAIL'}")
    IterBeta.clear_c(W.lib)
    print("ALL OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
