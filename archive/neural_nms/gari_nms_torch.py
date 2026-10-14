"""Differentiable PyTorch replica of ldpc_dec_msaa_quantum_serial_big_matrix_c (hbp_decoder_v2.c).

Same row-serial (layered) normalised min-sum schedule, but
  * batched over shots,
  * alpha replaced by a per-iteration beta_l,
  * gradients flow to beta_l through the soft output `lambda`.

Speed trick (exact): rows are visited in the C order, but rows that share no variable node commute, so
they are grouped into dependency levels and each level is updated in one vectorised step. The result is
identical to the strictly serial loop; the number of sequential steps drops from R rows to #levels.

Schedules: random_order = 0 -> deterministic order of the C code (custom_schedule 0..3).
           random_order = s > 0 -> the same xorshift32 row permutations the C code draws at iteration
           `loop` with seed s + loop (permutations accumulate across iterations exactly as in C).
           For the ensemble decoder, member `ens` uses s = random_order * ensemble_size + Nloop * ens.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_BIG = 1e30


# ----------------------------------------------------------------------------- row schedules
def xorshift_permute_(arr, seed):
    """In-place copy of random_permutation() from hbp_decoder_v2.c (xorshift32 Fisher-Yates)."""
    n = len(arr)
    if n <= 1:
        return
    state = int(seed) & 0xFFFFFFFF
    if state == 0:
        state = 1
    for i in range(n - 1, 0, -1):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        j = state % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]


class RowOrder:
    """Row order of every iteration, reproducing the C code. Call order(loop) for loop = 0, 1, 2, ..."""

    def __init__(self, R, hshape0, dxshape0, custom_schedule, random_order):
        self.cs, self.ro = int(custom_schedule), int(random_order)
        self.full = np.arange(R)
        self.bottom = np.arange(hshape0, R)
        self.ab = np.arange(hshape0)
        self.a = np.arange(dxshape0)
        self.b = np.arange(dxshape0, hshape0)

    def order(self, loop):
        cs, ro = self.cs, self.ro
        if cs == 0:
            if ro:
                xorshift_permute_(self.full, ro + loop)
            return self.full.copy()
        if ro:
            if cs == 1:
                xorshift_permute_(self.ab, ro + loop)
            else:  # cs 2 / 3: the two blocks use independent calls with the same seed
                xorshift_permute_(self.a, ro + loop)
                xorshift_permute_(self.b, ro + loop)
        if cs == 1:
            return np.concatenate([self.bottom, self.ab])
        if cs == 2:
            return np.concatenate([self.bottom, self.a, self.b])
        return np.concatenate([self.bottom, self.b, self.a])  # cs == 3


def build_row_order(R, hshape0, dxshape0, custom_schedule=0):
    """Deterministic (random_order == 0) order, kept for convenience."""
    return RowOrder(R, hshape0, dxshape0, custom_schedule, 0).order(0)


def level_schedule(Hrows, N, order):
    """Group rows into levels: rows in one level share no column, and rows sharing a column keep
    their relative order from `order`. Returns a list of int arrays of row indices."""
    R = Hrows.shape[0]
    valid = Hrows > 0
    last = np.zeros(N, dtype=np.int64)
    level = np.zeros(R, dtype=np.int64)
    rowcols = [Hrows[r][valid[r]].astype(np.int64) - 1 for r in range(R)]
    for r in order:
        cols = rowcols[r]
        lv = last[cols].max() + 1
        level[r] = lv
        last[cols] = lv
    srt = np.argsort(level, kind="stable")
    bounds = np.searchsorted(level[srt], np.arange(1, level.max() + 2))
    return [srt[bounds[i]:bounds[i + 1]] for i in range(level.max())]


# ----------------------------------------------------------------------------- the decoder
class GariNMSTorch(nn.Module):
    def __init__(self, Hrows, hshape0, dxshape0, N=None, custom_schedule=0, dtype=torch.float64,
                 max_cached_schedules=8):
        super().__init__()
        Hrows = np.asarray(Hrows, dtype=np.int64)
        self.Hrows_np = Hrows
        self.R, self.D = Hrows.shape
        self.N = int(N) if N is not None else int(Hrows.max())
        self.hshape0, self.dxshape0 = int(hshape0), int(dxshape0)
        self.cs = int(custom_schedule)
        self.dtype = dtype
        self.max_cached = max_cached_schedules
        assert self.D >= 2

        valid = np.cumprod(Hrows > 0, axis=1).astype(bool)           # C stops at the first 0
        assert ((Hrows > 0) == valid).all(), "Hrows must be zero-padded at the end of each row"
        assert (valid.sum(1) >= 2).all(), "every check needs degree >= 2"
        self.E = int(valid.sum())
        ptr = np.concatenate([[0], np.cumsum(valid.sum(1))])[:-1]
        eid = np.where(valid, ptr[:, None] + np.arange(self.D)[None, :], self.E)   # pad -> dummy edge E
        self._cols_np = np.where(valid, Hrows - 1, self.N)                          # pad -> dummy column N
        self._eid_np, self._valid_np = eid, valid
        self.register_buffer("cols_all", torch.from_numpy(self._cols_np))
        self.register_buffer("mask_all", torch.from_numpy(valid))
        self._cache = OrderedDict()   # random_order -> (RowOrder, [levels of loop 0, loop 1, ...])

    # -- schedules
    def _make_levels(self, order):
        out = []
        dev = self.cols_all.device
        for rows in level_schedule(self.Hrows_np, self.N, order):
            dl = int(self._valid_np[rows].sum(1).max())
            out.append((torch.from_numpy(rows).to(dev),
                        torch.from_numpy(self._cols_np[rows][:, :dl]).to(dev),
                        torch.from_numpy(self._eid_np[rows][:, :dl]).to(dev),
                        torch.from_numpy(self._valid_np[rows][:, :dl]).to(dev)))
        return out

    def levels_for(self, random_order, loop):
        ro = int(random_order)
        if ro not in self._cache:
            self._cache[ro] = (RowOrder(self.R, self.hshape0, self.dxshape0, self.cs, ro), [])
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        gen, lst = self._cache[ro]
        if ro == 0:
            if not lst:
                lst.append(self._make_levels(gen.order(0)))
            return lst[0]
        while len(lst) <= loop:
            lst.append(self._make_levels(gen.order(len(lst))))
        return lst[loop]

    @property
    def num_levels(self):
        return len(self.levels_for(0, 0))

    # -- one decoding iteration (pure function of its tensor inputs, so it can be checkpointed)
    def _iterate(self, lam, etar_in, beta, syn_f, levels):
        dt = lam.dtype
        lam_w = lam.clone()
        etar_new = torch.zeros_like(etar_in)
        for rows, cols, eid, mask in levels:
            t = lam_w[:, cols] - etar_in[:, eid]                          # variable -> check messages
            neg = (t < 0) & mask
            sign_t = 1.0 - 2.0 * (neg.sum(-1) % 2).to(dt)                 # product of signs, (B,G)
            absv = t.abs().masked_fill(~mask, _BIG)
            vals, idx = torch.topk(absv, 2, dim=-1, largest=False)
            is_pos = torch.zeros_like(absv, dtype=torch.bool).scatter_(-1, idx[..., :1], True)
            pr = torch.where(is_pos, vals[..., 1:2], vals[..., 0:1]) * beta
            sign_e = sign_t.unsqueeze(-1) * (1.0 - 2.0 * (t < 0).to(dt))
            etar = sign_e * pr * syn_f[:, rows].unsqueeze(-1)
            etar = torch.where(mask, etar, torch.zeros_like(etar))
            lam_w[:, cols] = torch.where(mask, t + etar, torch.zeros_like(t))
            etar_new[:, eid] = etar
        return lam_w, etar_new

    def forward(self, llr, syn, betas, Nloop, early_stopping=True, return_hist=False,
                random_order=0, checkpoint_iters=False):
        """
        llr    : (N,) or (B,N) prior LLRs
        syn    : (B,R) entries +1 / -1 (-1 = violated check), same convention as Syn_x in C
        betas  : float, or 1-D tensor; iteration `loop` uses betas[min(loop, len-1)]
        Nloop  : as in C, at most Nloop+1 iterations are run
        random_order : 0 = deterministic schedule, s>0 = the C code's random schedule with seed s
        checkpoint_iters : recompute each iteration in the backward pass (saves a lot of memory)
        returns dict(lam (B,N), hard (B,N) int8, iters (B,) as the python wrapper returns it,
                     converged (B,) bool, hist list of (B,N) or None)
        """
        dt = self.dtype
        dev = self.cols_all.device
        syn = torch.as_tensor(syn, device=dev)
        B = syn.shape[0]
        syn_f = syn.to(dt)
        s_bit = (1 - syn.to(torch.int64)) // 2                            # target parity per row
        llr = torch.as_tensor(llr, dtype=dt, device=dev).clamp(-1e4, 1e4)
        lam = torch.zeros(B, self.N + 1, dtype=dt, device=dev)
        lam[:, : self.N] = llr.expand(B, self.N)
        etar_old = torch.zeros(B, self.E + 1, dtype=dt, device=dev)
        if not torch.is_tensor(betas):
            betas = torch.tensor([float(betas)], dtype=dt, device=dev)
        betas = betas.to(device=dev, dtype=dt)

        chk = torch.arange(self.dxshape0, self.hshape0, device=dev) if early_stopping \
            else torch.arange(self.R, device=dev)
        chk_cols, chk_mask, chk_s = self.cols_all[chk], self.mask_all[chk], s_bit[:, chk]

        active = torch.ones(B, dtype=torch.bool, device=dev)
        iters = torch.full((B,), Nloop + 1, dtype=torch.int64, device=dev)
        hist = [] if return_hist else None

        for loop in range(Nloop + 1):
            beta = betas[min(loop, betas.shape[0] - 1)]
            levels = self.levels_for(random_order, loop)
            if checkpoint_iters and torch.is_grad_enabled():
                lam_w, etar_new = checkpoint(lambda a, b, c, lv=levels: self._iterate(a, b, c, syn_f, lv),
                                             lam, etar_old, beta, use_reentrant=False)
            else:
                lam_w, etar_new = self._iterate(lam, etar_old, beta, syn_f, levels)

            with torch.no_grad():
                hard = lam_w[:, : self.N] < 0
                hard_p = torch.cat([hard, torch.zeros(B, 1, dtype=torch.bool, device=dev)], dim=1)
                par = (hard_p[:, chk_cols] & chk_mask).sum(-1) % 2
                conv = (par == chk_s).all(dim=1)

            lam = torch.where(active[:, None], lam_w, lam)                 # converged shots stay frozen
            etar_old = torch.where(active[:, None], etar_new, etar_old)
            iters = torch.where(active & conv, torch.full_like(iters, loop), iters)
            active = active & ~conv
            if return_hist:
                hist.append(lam[:, : self.N])
            if not bool(active.any()):
                break

        lam_out = lam[:, : self.N]
        return {"lam": lam_out,
                "hard": (lam_out < 0).to(torch.int8),
                "iters": iters.clamp(max=Nloop),
                "converged": iters <= Nloop,
                "hist": hist}
