"""Torch-free problem construction + DEM sampler (same code as in train_beta_v4.py)."""

import argparse
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import stim
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

from helper_functions import bb_circuit, det_and_err_encoding, build_Hcols_Hrows


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

