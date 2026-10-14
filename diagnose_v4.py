"""Step-1 diagnostics for GARI-NMS (C decoder only, no torch needed).

 (1) headroom : LER vs scalar alpha on the SAME shots (paired against --base_alpha)
 (2) taxonomy : every logical failure is either
        - 'wrong class' : BP converged (all checks satisfied) but to a correction in the wrong logical class
        - 'no converge' : BP hit the iteration cap. NOTE: the C ensemble call then returns an all-zero correction,
                          so such a shot counts as a failure only if the true observable is flipped.
Run from the repo root:
    python diagnose_v4.py --d 6 --p 0.004 --shots 262144 --threads 8
    python diagnose_v4.py --d 6 --p 0.002 0.001 --shots 1000000 --alphas 0.9 0.96875 1.0 --threads 8
"""
import argparse
import time
import numpy as np
from scipy.stats import binomtest

from nms_problem_v4 import build_problem, Sampler
from my_decoders import hbplib_wrapper_v4 as W


def decode(prob, syn, obs, alpha, mitr, rs, cs, et, threads):
    le, it = W.ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler(
        prob.llr, prob.llr_ab, syn.shape[0], mitr, prob.Hrows, prob.Hcols, alpha, syn,
        rs, prob.matrices.check_matrix.shape, prob.A.dx.shape, 1, 'i',
        prob.nx, prob.nz, prob.mx, prob.m, prob.A.dz, prob.l_dz, cs,
        early_stopping=bool(et), num_threads=threads)
    return np.any((le + obs) % 2, axis=1), np.asarray(it)


def pct(a, b):
    return 100.0 * a / max(b, 1)


def run_one_p(a, p):
    prob = build_problem(a.d, p, a.prt)
    smp = Sampler(prob, seed=a.seed)
    alphas = list(dict.fromkeys([a.base_alpha] + list(a.alphas)))
    fails = {x: [] for x in alphas}
    iters = {x: [] for x in alphas}
    obsflip = []
    done, t0 = 0, time.time()
    while done < a.shots:
        k = min(a.chunk, a.shots - done)
        syn, _, obs = smp.sample(k, need_target=False)
        obsflip.append(np.any(obs, axis=1))
        for x in alphas:
            f, it = decode(prob, syn, obs, x, a.mitr, a.rs, a.cs, a.et, a.threads)
            fails[x].append(f); iters[x].append(it)
        done += k
        print(f"  p={p}: {done}/{a.shots} shots  ({time.time() - t0:.0f}s)", flush=True)
    fails = {x: np.concatenate(v) for x, v in fails.items()}
    iters = {x: np.concatenate(v) for x, v in iters.items()}
    obsflip = np.concatenate(obsflip)
    n = a.shots
    base = fails[a.base_alpha]

    print("\n" + "=" * 96)
    print(f"d={a.d}  p={p}  shots={n}  mitr={a.mitr}  cs={a.cs}  rs={a.rs}   (same shots for every alpha)")
    print("=" * 96)
    print("(1) HEADROOM: logical error rate vs scalar alpha")
    print(f"{'alpha':>9} {'fails':>7} {'LER':>10} {'+-':>9} {'avg_it':>7} | vs alpha={a.base_alpha}: only_base only_this  net   p-value")
    for x in alphas:
        f = fails[x]
        ob, ot = int((base & ~f).sum()), int((~base & f).sum())
        pv = binomtest(ot, ob + ot, 0.5).pvalue if ob + ot > 0 else 1.0
        tag = "  <- base" if x == a.base_alpha else ""
        print(f"{x:9.5f} {int(f.sum()):7d} {f.mean():10.5f} {np.sqrt(max(f.sum(), 1)) / n:9.5f} "
              f"{iters[x].mean():7.2f} | {ob:20d} {ot:9d} {ob - ot:+5d}   {pv:.3g}{tag}")
    best = min(alphas, key=lambda x: fails[x].sum())
    gain = 100 * (fails[a.base_alpha].sum() - fails[best].sum()) / max(fails[a.base_alpha].sum(), 1)
    print(f"best scalar alpha on these shots: {best} ({gain:+.1f}% fewer failures than {a.base_alpha}; "
          f"picking the best of {len(alphas)} on the same shots is optimistic)")

    print("\n(2) FAILURE TAXONOMY (alpha = %s)" % a.base_alpha)
    it = iters[a.base_alpha]
    nonconv = it >= a.mitr
    wrong = base & ~nonconv
    cap = base & nonconv
    print(f"  shots decoded              : {n}")
    print(f"  true observable flipped    : {int(obsflip.sum())}  ({pct(obsflip.sum(), n):.2f}%)  <- failure rate of 'do nothing'")
    print(f"  total logical failures     : {int(base.sum())}")
    print(f"    wrong class (converged)  : {int(wrong.sum()):7d}  ({pct(wrong.sum(), base.sum()):5.1f}% of failures)")
    print(f"    hit iteration cap        : {int(cap.sum()):7d}  ({pct(cap.sum(), base.sum()):5.1f}% of failures)")
    print(f"  shots hitting iteration cap: {int(nonconv.sum())}  ({pct(nonconv.sum(), n):.3f}% of shots); "
          f"of those {int((nonconv & ~base).sum())} count as success because the observable was not flipped")
    if wrong.any():
        q = np.percentile(it[wrong], [25, 50, 75, 90])
        print(f"  iterations of wrong-class failures  (25/50/75/90%): {q.round(0)}")
    ok = ~base & ~nonconv
    if ok.any():
        q = np.percentile(it[ok], [25, 50, 75, 90])
        print(f"  iterations of correct converged shots (25/50/75/90%): {q.round(0)}")
    print("=" * 96)
    return dict(p=p, alphas=alphas, fails={x: int(fails[x].sum()) for x in alphas},
                wrong=int(wrong.sum()), cap=int(cap.sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--p", type=float, nargs="+", default=[0.004])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.875, 0.90, 0.9375, 0.96875, 1.0])
    ap.add_argument("--base_alpha", type=float, default=0.96875)
    ap.add_argument("--shots", type=int, default=262144)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--mitr", type=int, default=400)
    ap.add_argument("--cs", type=int, default=2)
    ap.add_argument("--rs", type=int, default=1)
    ap.add_argument("--et", type=int, default=1)
    ap.add_argument("--prt", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2024, help="shot seed (a new one gives unseen shots)")
    a = ap.parse_args()
    for p in a.p:
        run_one_p(a, p)


if __name__ == "__main__":
    main()
