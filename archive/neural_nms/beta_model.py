"""Iteration-wise learnable normalisation factors beta_l for GARI-NMS (Neural-NMS, Step 1)."""
import json
import math
import numpy as np
import torch
import torch.nn as nn


class IterBeta(nn.Module):
    """beta_l for l = 0..num_iters-1, constrained to (0, beta_max).

    num_free < num_iters  ->  only the first `num_free` betas are free; later iterations
    reuse the last free one (fewer parameters / less storage in hardware).
    quant_bits > 0        ->  forward pass uses beta rounded to a multiple of 2**-quant_bits
    (straight-through gradient), for hardware-aware fine-tuning.
    """

    def __init__(self, num_iters, init=0.9375, num_free=None, beta_max=1.0, quant_bits=0):
        super().__init__()
        assert 0.0 < init < beta_max
        self.num_iters = int(num_iters)
        self.num_free = int(num_free) if num_free else self.num_iters
        self.beta_max = float(beta_max)
        self.quant_bits = int(quant_bits)
        p = init / beta_max
        self.raw = nn.Parameter(torch.full((self.num_free,), math.log(p / (1.0 - p)), dtype=torch.float64))

    def free_betas(self):
        b = self.beta_max * torch.sigmoid(self.raw)
        if self.quant_bits > 0:
            s = float(2 ** self.quant_bits)
            bq = torch.round(b * s) / s
            b = b + (bq - b).detach()  # straight-through estimator
        return b

    def forward(self):
        """Tensor of shape (num_iters,)."""
        b = self.free_betas()
        if self.num_free < self.num_iters:
            pad = b[-1:].expand(self.num_iters - self.num_free)
            b = torch.cat([b, pad])
        return b

    def beta_at(self, loop):
        return self.forward()[min(loop, self.num_iters - 1)]

    # ---- export / import ------------------------------------------------------------
    def to_numpy(self):
        with torch.no_grad():
            return self.forward().cpu().numpy().astype(np.float64)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"num_iters": self.num_iters, "num_free": self.num_free,
                       "beta_max": self.beta_max, "quant_bits": self.quant_bits,
                       "raw": self.raw.detach().cpu().tolist(),
                       "beta": self.to_numpy().tolist()}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["num_iters"], num_free=d["num_free"], beta_max=d["beta_max"], quant_bits=d["quant_bits"])
        with torch.no_grad():
            m.raw.copy_(torch.tensor(d["raw"], dtype=torch.float64))
        return m

    def push_to_c(self, lib):
        """Make the C decoder use these betas (one per iteration). lib = ctypes.CDLL of hbplib_v2.so
        (e.g. hbplib_wrapper_v2.lib). Call clear_c(lib) to go back to the scalar alpha."""
        import ctypes
        arr = np.ascontiguousarray(self.to_numpy(), dtype=np.float64)
        lib.hbp_set_alpha_table.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.hbp_set_alpha_table.restype = None
        lib.hbp_set_alpha_table(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(len(arr)))

    @staticmethod
    def clear_c(lib):
        import ctypes
        lib.hbp_set_alpha_table.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.hbp_set_alpha_table.restype = None
        lib.hbp_set_alpha_table(None, 0)
