"""Neural-NMS step 1: train iteration-wise beta_l for GARI-NMS on circuit-level BB-code data.

Run from the repo root (same place as stim_batched_data_v2.py):
    python train_beta_v4.py --d 6 --p 0.002 --steps 200 --batch 256 --train_iters 30

What it does
  1. rebuilds the problem exactly like stim_batched_data_v2.py (circuit, DEM, big matrix, priors, llr),
  2. checks the torch decoder against the C decoder on real shots,
  3. samples (syndrome, true error) pairs from the DEM and trains beta_0..beta_L with Adam,
  4. loads the learned betas into the C decoder (hbp_set_alpha_table) and compares its logical error
     rate / iterations with the fixed-alpha baseline on fresh shots, using the same batched ensemble call
     as stim_batched_data_v2.py.
"""
import argparse
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import stim
import torch
import torch.nn.functional as F
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

from helper_functions import bb_circuit, det_and_err_encoding, build_Hcols_Hrows
from my_decoders import hbplib_wrapper_v4 as W
from neural_nms_v4 import IterBeta, GariNMSTorch


# --------------------------------------------------------------------------------------- problem
@dataclass
class Problem:
    d: int
    p: float
    code: object
    dem: stim.DetectorErrorModel
    matrices: object          # check_matrix, observables_matrix, priors
    A: object                 # data_encoding (dx, dz, i_h*_only, det_index, big_matrix)
    Hrows: np.ndarray
    Hcols: np.ndarray
    llr: np.ndarray
    llr_ab: np.ndarray
    l_dz: np.ndarray
    nx: int
    nz: int
    mx: int
    m: int
    R: int                    # rows of the big matrix
    N: int                    # columns of the big matrix


def build_problem(d, p, prt=0, data_dir="data/circuits"):
    """Same construction as main() in stim_batched_data_v2.py (CL noise, z basis, use_both)."""
    assert d % 2 == 0, "only the bivariate-bicycle branch is supported here"
    noise_tag = "CL_both"
    os.makedirs(data_dir, exist_ok=True)
    _, code = bb_circuit(d, p=0, p1=0, p2=0, p3=0, p4=0, r=d)
    fname = f"{data_dir}/{code.name}_d{d}_{noise_tag}_{p}.stim"
    try:
        circuit = stim.Circuit.from_file(fname)
    except Exception:
        circuit, code = bb_circuit(d, p, p1=p, p2=p, p3=p, p4=p, r=d, z_basis=True, use_both=True,
                                   HZH=False, use_hperp=False)
        with open(fname, "w") as f:
            circuit.to_file(f)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True,
                                       ignore_decomposition_failures=True)
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)

    mpkl = f"{data_dir}/{code.name}_d{d}_{noise_tag}_matrices.pkl"
    try:
        A = pickle.load(open(mpkl, "rb"))
    except Exception:
        A = det_and_err_encoding(d, matrices, circuit)
        pickle.dump(A, open(mpkl, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    hpkl = f"{data_dir}/{code.name}_d{d}_{noise_tag}_hcols_hrows.pkl"
    try:
        Hcols, Hrows = pickle.load(open(hpkl, "rb"))
    except Exception:
        Hcols, Hrows = build_Hcols_Hrows(A.big_matrix.toarray())
        pickle.dump((Hcols, Hrows), open(hpkl, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    h = matrices.check_matrix
    m, n = h.shape
    mx, nx = A.dx.shape
    mz, nz = A.dz.shape
    big = A.big_matrix
    l = matrices.observables_matrix.toarray().astype(np.int8)
    l_dz = l[:, A.i_hz_only]

    # priors / llr: copied from stim_batched_data_v2.py
    priors_big = np.zeros(big.shape[1], dtype=float)
    priors_ab = np.zeros(nx + nz, dtype=float)
    for i in range(nx):
        priors_ab[i] = np.sum(matrices.priors[np.where(A.i_dx_in_hx == i)[0]])
    for i in range(nz):
        priors_ab[nx + i] = np.sum(matrices.priors[np.where(A.i_dz_in_hz == i)[0]])
    if prt == 0:
        priors_big[: nx + nz] = 0.5
    elif prt == 2:
        priors_big[: nx + nz] = priors_ab
    else:
        raise NotImplementedError
    priors_big[nx + nz: 2 * nx + nz] = matrices.priors[A.i_hx_only]
    priors_big[2 * nx + nz: 2 * (nx + nz)] = matrices.priors[A.i_hz_only]
    priors_big[2 * (nx + nz):] = matrices.priors[A.i_hy_only]
    llr = np.log((1 - priors_big) / priors_big)
    llr_ab = np.log((1 - priors_ab) / priors_ab)
    return Problem(d, p, code, dem, matrices, A, Hrows, Hcols, llr, llr_ab, l_dz, nx, nz, mx, m,
                   big.shape[0], big.shape[1])


# --------------------------------------------------------------------------------------- data
class Sampler:
    """Samples (syndrome, true big-matrix variables, observables) from the DEM."""

    def __init__(self, prob, seed=0):
        self.prob = prob
        self.sampler = prob.dem.compile_sampler(seed=seed)
        A = prob.A
        big = A.big_matrix.tocsr()
        self.big = big
        self.bottom = big[prob.m:, prob.nx + prob.nz:].astype(np.int32).tocsr()   # tie rows: var_i = sum(rest)
        self.col_order = np.concatenate([A.i_hx_only, A.i_hz_only, A.i_hy_only])
        self.det_x = np.where(A.det_index == 1)[0]
        self.det_z = np.where(A.det_index == 3)[0]
        self._checked = False

    def syndrome(self, det):
        p = self.prob
        s = np.zeros((det.shape[0], p.R), dtype=np.int8)
        s[:, : p.mx] = det[:, self.det_x]
        s[:, p.mx: p.m] = det[:, self.det_z]
        return (1 - 2 * s).astype(np.int8)                      # +1 / -1, as in stim_batched_data_v2.py

    def sample(self, shots, need_target=True):
        p = self.prob
        det, obs, err = self.sampler.sample(shots=shots, return_errors=True, bit_packed=False)
        det, obs = det.astype(np.int8), obs.astype(np.int8)
        if not need_target:
            return self.syndrome(det), None, obs
        err = err.astype(np.int8)
        rest = err[:, self.col_order]                            # individual mechanisms, big-matrix order
        top = (self.bottom @ rest.T.astype(np.int32)).T % 2      # 'total' variables
        target = np.concatenate([top.astype(np.int8), rest], axis=1)
        if not self._checked:                                    # the DEM column order must match the matrices
            lhs = (self.big @ target.T.astype(np.int32)).T % 2
            rhs = np.zeros_like(lhs)
            rhs[:, : p.m] = (1 - self.syndrome(det)[:, : p.m]) // 2
            assert (lhs == rhs).all(), "sampled errors do not reproduce the syndrome: column order mismatch"
            assert ((err @ p.matrices.check_matrix.toarray().T.astype(np.int32)) % 2 == det).all()
            self._checked = True
        return self.syndrome(det), target, obs


# --------------------------------------------------------------------------------------- C decoder
def eval_c(prob, syn, obs, alpha, mitr, rs, cs, et, ens, poi, threads):
    """Exactly the batched call used in stim_batched_data_v2.py."""
    t = time.time()
    le, it = W.ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler(
        prob.llr, prob.llr_ab, syn.shape[0], mitr, prob.Hrows, prob.Hcols, alpha, syn,
        rs, prob.matrices.check_matrix.shape, prob.A.dx.shape, ens, poi,
        prob.nx, prob.nz, prob.mx, prob.m, prob.A.dz, prob.l_dz, cs,
        early_stopping=bool(et), num_threads=threads)
    fail = np.any((le + obs) % 2, axis=1)
    return fail, np.asarray(it), time.time() - t


def decode_fixed(prob, syn, obs, a):
    """Baseline: scalar alpha (= --init), the learned table switched off."""
    IterBeta.clear_c(W.lib)
    return eval_c(prob, syn, obs, a.init, a.eval_mitr, a.rs, a.cs, a.et, a.eval_ens, a.eval_poi, a.threads)


def decode_beta(prob, syn, obs, a, bm):
    bm.push_to_c(W.lib)
    try:
        return eval_c(prob, syn, obs, a.init, a.eval_mitr, a.rs, a.cs, a.et, a.eval_ens, a.eval_poi, a.threads)
    finally:
        IterBeta.clear_c(W.lib)                     # never leave the global table switched on


def compare_on_fresh(prob, bm, a, shots, seed=12345, chunk=16384):
    """Decode the SAME fresh shots with fixed alpha and with the learned betas; paired statistics."""
    from scipy.stats import binomtest
    smp = Sampler(prob, seed=seed)
    fb, fl, ib, il = [], [], [], []
    done = 0
    while done < shots:
        k = min(chunk, shots - done)
        syn, _, obs = smp.sample(k, need_target=False)
        f0, i0, _ = decode_fixed(prob, syn, obs, a)
        f1, i1, _ = decode_beta(prob, syn, obs, a, bm)
        fb.append(f0); fl.append(f1); ib.append(i0); il.append(i1)
        done += k
        print(f"  eval {done}/{shots}: fails fixed={int(np.concatenate(fb).sum())} learned={int(np.concatenate(fl).sum())}", flush=True)
    fb, fl, ib, il = map(np.concatenate, (fb, fl, ib, il))
    both, only_b, only_l = int((fb & fl).sum()), int((fb & ~fl).sum()), int((~fb & fl).sum())
    nd = only_b + only_l
    pval = binomtest(only_l, nd, 0.5).pvalue if nd > 0 else 1.0
    print("=" * 78)
    print(f"shots={shots}  p={prob.p}  mitr={a.eval_mitr}")
    print(f"fixed alpha={a.init:<8} LER={fb.mean():.5f} +- {np.sqrt(max(fb.sum(),1))/shots:.5f}  fails={int(fb.sum())}  avg_iters={ib.mean():.3f}")
    print(f"learned beta_l       LER={fl.mean():.5f} +- {np.sqrt(max(fl.sum(),1))/shots:.5f}  fails={int(fl.sum())}  avg_iters={il.mean():.3f}")
    print(f"paired: both fail={both}  only fixed fails={only_b}  only learned fails={only_l}")
    print(f"net gain (fixed - learned) = {only_b - only_l:+d} shots;  exact two-sided binomial p = {pval:.3g}"
          f"  (p<0.05 -> difference unlikely to be noise)")
    print("=" * 78)
    return fb, fl


def check_against_c(prob, dec, sampler, bm, Nloop, ro, cs, et, shots=16):
    syn, _, _ = sampler.sample(shots)
    bm.push_to_c(W.lib)
    with torch.no_grad():
        r = dec(prob.llr, syn, bm().detach(), Nloop, early_stopping=bool(et), random_order=ro)
    worst, same_it = 0.0, True
    for k in range(shots):
        hd, it, lam = W.ldpc_dec_msaa_quantum_serial_big_matrix_c(
            prob.llr, Nloop, prob.Hrows, prob.Hcols, 0.5, syn[k], ro, prob.matrices.check_matrix.shape,
            prob.A.dx.shape, cs, bool(et))
        worst = max(worst, float(np.abs(r["lam"][k].detach().cpu().numpy() - lam).max()))
        same_it &= int(r["iters"][k]) == int(it)
    IterBeta.clear_c(W.lib)
    return worst, same_it


# --------------------------------------------------------------------------------------- training
def build_hard_bank(prob, a, n_t, seed):
    """Decode fresh shots with the fixed-alpha C decoder and keep the hard ones:
    logical failure, or needing >= --hard_iters iterations. Returns (syn, [target[:, :n_t] | observables])."""
    smp = Sampler(prob, seed=seed)
    syns, tgts = [], []
    tot = n_fail = n_slow = 0
    while tot < a.bank_shots:
        k = min(16384, a.bank_shots - tot)
        syn, tgt, obs = smp.sample(k)
        fail, it, _ = decode_fixed(prob, syn, obs, a)
        slow = it >= a.hard_iters
        hard = fail | slow
        syns.append(syn[hard]); tgts.append(np.concatenate([tgt[hard][:, :n_t], obs[hard]], axis=1))
        tot += k; n_fail += int(fail.sum()); n_slow += int(slow.sum())
        print(f"  bank: scanned {tot}/{a.bank_shots}, hard so far {sum(len(x) for x in syns)} "
              f"(logical fails {n_fail}, slow>={a.hard_iters} its {n_slow})", flush=True)
    return np.concatenate(syns), np.concatenate(tgts)


def weighted_bce(hist, tgt_t, n_t, w_it):
    """BCE of every iteration's soft output against the true signature variables."""
    return sum(w_it[l] * F.binary_cross_entropy_with_logits(-hist[l][:, :n_t], tgt_t[:, :n_t]) for l in range(len(hist)))


def soft_logical_parity(lam_z, Lmat):
    """P(logical bit = 1) for each observable from the soft outputs of the Z-correction variables.
    P(x_j = 1) = sigmoid(-lam_j)  ->  E[(-1)^x_j] = tanh(lam_j/2);  parity prob = (1 - prod_j tanh(lam_j/2)) / 2."""
    t = torch.tanh(lam_z / 2)
    logabs = torch.log(t.abs().clamp_min(1e-12))
    s = logabs @ Lmat.T                                   # (B, n_obs): log |product|
    neg = ((t < 0).to(lam_z.dtype) @ Lmat.T) % 2          # parity of the number of negative factors
    prod = (1.0 - 2.0 * neg) * torch.exp(s)
    return ((1.0 - prod) / 2).clamp(1e-7, 1 - 1e-7)


def weighted_logical(hist, obs_t, Lmat, nx, nz, w_it):
    """BCE between the soft logical parity and the true observable flip, summed over iterations."""
    tot = 0.0
    for l in range(len(hist)):
        pl = soft_logical_parity(hist[l][:, nx: nx + nz], Lmat)
        tot = tot + w_it[l] * F.binary_cross_entropy(pl, obs_t)
    return tot


def total_loss(hist, tgt_t, n_t, w_it, a, Lmat, nx, nz):
    if a.loss == "sig":
        return weighted_bce(hist, tgt_t, n_t, w_it)
    lg = weighted_logical(hist, tgt_t[:, n_t:], Lmat, nx, nz, w_it)
    if a.loss == "logical":
        return lg
    return weighted_bce(hist, tgt_t, n_t, w_it) + a.logical_weight * lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--p_train", type=float, default=0.0,
                    help="noise rate used to sample TRAINING shots/llr (0 = same as --p). Evaluation always uses --p.")
    ap.add_argument("--prt", type=int, default=0, help="prior type, as in stim_batched_data_v2.py")
    ap.add_argument("--cs", type=int, default=2, help="custom_random_schedule_HBP")
    ap.add_argument("--rs", type=int, default=1, help="random_order seed of the decoder")
    ap.add_argument("--et", type=int, default=1, help="early stopping")
    ap.add_argument("--train_iters", type=int, default=30, help="unrolled iterations during training (Nloop)")
    ap.add_argument("--num_free", type=int, default=0, help="# free betas (0 = train_iters+1); later iterations reuse the last")
    ap.add_argument("--init", type=float, default=0.96875, help="initial beta (= the fixed msf of the baseline)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lr_final_frac", type=float, default=0.1, help="cosine decay of lr down to lr*this")
    ap.add_argument("--hard_frac", type=float, default=0.5, help="fraction of each batch drawn from the hard-shot bank (0 = off)")
    ap.add_argument("--bank_shots", type=int, default=100000, help="# shots scanned to build the hard-shot bank")
    ap.add_argument("--hard_iters", type=int, default=20, help="a shot is 'hard' if the fixed-alpha C decoder fails or needs >= this many iterations")
    ap.add_argument("--val_shots", type=int, default=256, help="size of the fixed validation set")
    ap.add_argument("--val_every", type=int, default=20)
    ap.add_argument("--n_schedules", type=int, default=1, help="# different random schedules cycled through in training")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--checkpoint", type=int, default=1)
    ap.add_argument("--loss_vars", default="total", choices=["total", "all"],
                    help="total: supervise the first nx+nz (signature) variables; all: also the individual mechanisms")
    ap.add_argument("--quant_bits", type=int, default=0)
    ap.add_argument("--groups", type=int, default=1, choices=[1, 3],
                    help="1: one beta per iteration; 3: separate beta per iteration for A rows / B rows / tie rows")
    ap.add_argument("--loss", default="sig", choices=["sig", "logical", "both"],
                    help="sig: BCE vs true signature variables (old); logical: soft parity of the logical operators vs the true "
                         "observable flip; both: sig + --logical_weight * logical")
    ap.add_argument("--logical_weight", type=float, default=1.0)
    ap.add_argument("--eval_shots", type=int, default=65536)
    ap.add_argument("--eval_mitr", type=int, default=400)
    ap.add_argument("--eval_ens", type=int, default=1)
    ap.add_argument("--eval_poi", default="i")
    ap.add_argument("--eval_seed", type=int, default=12345, help="seed of the evaluation shots (use a new one for an independent test set)")
    ap.add_argument("--eval_beta", default="", help="skip training: evaluate this saved beta json against fixed alpha")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="beta_v4_d{d}_p{p}.json")
    ap.add_argument("--no_check", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dt = torch.float64 if a.dtype == "float64" else torch.float32
    out = a.out.format(d=a.d, p=a.p)

    prob = build_problem(a.d, a.p, a.prt)
    print(f"big matrix {prob.R} x {prob.N}, nx={prob.nx} nz={prob.nz} mx={prob.mx} m={prob.m}, "
          f"edges={int((prob.Hrows > 0).sum())}, max row degree={prob.Hrows.shape[1]}")

    if a.eval_beta:                                                   # evaluation only
        bm = IterBeta.load(a.eval_beta)
        print(f"loaded {a.eval_beta}: {bm.num_iters} iterations x {bm.num_groups} group(s), first {bm.to_numpy()[:2].round(4).tolist()}, last {np.atleast_1d(bm.to_numpy()[-1]).round(4).tolist()}")
        compare_on_fresh(prob, bm, a, a.eval_shots, seed=a.eval_seed)
        return

    dec = GariNMSTorch(prob.Hrows, prob.m, prob.mx, N=prob.N, custom_schedule=a.cs, dtype=dt,
                       max_cached_schedules=max(8, a.n_schedules)).to(a.device)
    seeds = [a.rs + 1000 * k for k in range(a.n_schedules)]
    L = a.train_iters + 1
    bm = IterBeta(L, init=a.init, num_free=a.num_free or L, quant_bits=a.quant_bits, num_groups=a.groups).to(a.device)
    Lmat = torch.from_numpy(prob.l_dz.astype(np.float64)).to(device=a.device, dtype=dt)
    tprob = prob if (not a.p_train or a.p_train == a.p) else build_problem(a.d, a.p_train, a.prt)
    assert tprob.Hrows.shape == prob.Hrows.shape and (tprob.Hrows == prob.Hrows).all(), "graph differs between p and p_train"
    sampler = Sampler(tprob, seed=a.seed + 1)
    check_sampler = Sampler(prob, seed=a.seed + 2)
    n_t = tprob.nx + tprob.nz if a.loss_vars == "total" else tprob.N
    print(f"levels per iteration (deterministic schedule): {dec.num_levels}")

    if not a.no_check:
        w, same = check_against_c(prob, dec, check_sampler, bm, a.train_iters, a.rs, a.cs, a.et)
        print(f"torch vs C on real shots: max|dlam|={w:.2e}, iterations equal={same}")
        assert w < 1e-6 and same, "torch decoder does not match the C decoder on the real matrix"

    # ---- hard-shot bank and a fixed validation set
    bank_syn = bank_tgt = None
    n_val_hard = 0
    if a.hard_frac > 0:
        print(f"building hard-shot bank from {a.bank_shots} shots at p={tprob.p} ...")
        hs, ht = build_hard_bank(tprob, a, n_t, seed=a.seed + 7)
        if len(hs) >= 20:
            perm = rng.permutation(len(hs))
            n_val_hard = min(len(hs) // 10, a.val_shots // 2)
            vi, ti = perm[:n_val_hard], perm[n_val_hard:]
            val_hs, val_ht = hs[vi], ht[vi]
            bank_syn, bank_tgt = hs[ti], ht[ti]
            print(f"hard bank: {len(bank_syn)} train shots, {n_val_hard} held out for validation")
        else:
            print(f"WARNING: only {len(hs)} hard shots found -> hard mining disabled (raise --p_train / --bank_shots, lower --hard_iters)")
    vs, vt, vobs = Sampler(tprob, seed=a.seed + 3).sample(a.val_shots - n_val_hard)
    if n_val_hard:
        vs, vt = np.concatenate([val_hs, vs]), np.concatenate([val_ht, np.concatenate([vt[:, :n_t], vobs], axis=1)])
    else:
        vt = np.concatenate([vt[:, :n_t], vobs], axis=1)
    vkeep = (vs[:, : prob.m] < 0).any(1)
    val_syn = torch.from_numpy(vs[vkeep]).to(a.device)
    val_tgt = torch.from_numpy(vt[vkeep]).to(device=a.device, dtype=dt)
    print(f"validation set: {len(val_syn)} nontrivial shots ({n_val_hard} hard + fresh)")

    opt = torch.optim.Adam(bm.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps, 1), eta_min=a.lr * a.lr_final_frac)
    w_it = torch.arange(1, L + 1, dtype=dt, device=a.device)
    w_it = w_it / w_it.sum()                                          # later iterations weigh more
    t0 = time.time()
    for step in range(a.steps):
        if step % a.val_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                rv = dec(tprob.llr, val_syn, bm(), a.train_iters, early_stopping=bool(a.et), return_hist=True,
                         random_order=seeds[0])
                vloss = float(total_loss(rv["hist"], val_tgt, n_t, w_it, a, Lmat, prob.nx, prob.nz))
            b = bm.to_numpy().reshape(bm.num_iters, -1).mean(1)      # mean over groups, per iteration
            print(f"[val ] step {step:4d} val_loss={vloss:.5f} val_conv={float(rv['converged'].double().mean()):.3f} "
                  f"beta[0,1,2,-1]={b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
        n_hard = int(round(a.hard_frac * a.batch)) if bank_syn is not None else 0
        syn, tgt, tobs = sampler.sample(a.batch - n_hard) if a.batch - n_hard > 0 else (None, None, None)
        tgt = np.concatenate([tgt[:, :n_t], tobs], axis=1) if tgt is not None else None
        if n_hard:
            idx = rng.integers(0, len(bank_syn), n_hard)
            syn = bank_syn[idx] if syn is None else np.concatenate([bank_syn[idx], syn])
            tgt = bank_tgt[idx] if tgt is None else np.concatenate([bank_tgt[idx], tgt])
        keep = (syn[:, : prob.m] < 0).any(1)                          # skip trivial (empty-syndrome) shots
        if keep.sum() == 0:
            continue
        syn_t = torch.from_numpy(syn[keep]).to(a.device)
        tgt_t = torch.from_numpy(tgt[keep]).to(device=a.device, dtype=dt)
        r = dec(tprob.llr, syn_t, bm(), a.train_iters, early_stopping=bool(a.et), return_hist=True,
                random_order=seeds[step % len(seeds)], checkpoint_iters=bool(a.checkpoint))
        loss = total_loss(r["hist"], tgt_t, n_t, w_it, a, Lmat, prob.nx, prob.nz)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bm.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == a.steps - 1:
            print(f"[train] step {step:4d} loss={float(loss.detach()):.5f} shots={int(keep.sum())} (hard {n_hard}) "
                  f"conv={float(r['converged'].double().mean()):.3f}", flush=True)
    bm.save(out)
    print(f"saved {out}")
    compare_on_fresh(prob, bm, a, a.eval_shots, seed=a.eval_seed)


if __name__ == "__main__":
    main()
