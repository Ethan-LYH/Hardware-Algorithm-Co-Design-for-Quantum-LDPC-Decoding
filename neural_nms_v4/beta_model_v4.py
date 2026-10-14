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

    def __init__(self, num_iters, init=0.9375, num_free=None, beta_max=1.0, quant_bits=0, num_groups=1):
        super().__init__()
        assert 0.0 < init < beta_max
        self.num_iters = int(num_iters)
        self.num_free = int(num_free) if num_free else self.num_iters
        self.beta_max = float(beta_max)
        self.quant_bits = int(quant_bits)
        assert num_groups in (1, 3)
        self.num_groups = int(num_groups)      # 3 = separate beta for A rows, B rows, tie rows
        p = init / beta_max
        shape = (self.num_free,) if self.num_groups == 1 else (self.num_free, self.num_groups)
        self.raw = nn.Parameter(torch.full(shape, math.log(p / (1.0 - p)), dtype=torch.float64))

    def free_betas(self):
        b = self.beta_max * torch.sigmoid(self.raw)
        if self.quant_bits > 0:
            s = float(2 ** self.quant_bits)
            bq = torch.round(b * s) / s
            b = b + (bq - b).detach()  # straight-through estimator
        return b

    def forward(self):
        """Tensor of shape (num_iters,) or, with num_groups=3, (num_iters, 3)."""
        b = self.free_betas()
        if self.num_free < self.num_iters:
            pad = b[-1:].expand(self.num_iters - self.num_free, *b.shape[1:])
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
                       "beta_max": self.beta_max, "quant_bits": self.quant_bits, "num_groups": self.num_groups,
                       "raw": self.raw.detach().cpu().tolist(),
                       "beta": self.to_numpy().tolist()}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["num_iters"], num_free=d["num_free"], beta_max=d["beta_max"], quant_bits=d["quant_bits"],
                num_groups=d.get("num_groups", 1))
        with torch.no_grad():
            m.raw.copy_(torch.tensor(d["raw"], dtype=torch.float64))
        return m

    def push_to_c(self, lib):
        """Make the C decoder use these betas (one per iteration). lib = ctypes.CDLL of hbplib_v4.so
        (e.g. hbplib_wrapper_v4.lib). Call clear_c(lib) to go back to the scalar alpha."""
        import ctypes
        arr = np.ascontiguousarray(self.to_numpy(), dtype=np.float64)   # (L,) or (L,G), row-major = table[loop*G+g]
        lib.hbp_set_alpha_table_g.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
        lib.hbp_set_alpha_table_g.restype = None
        lib.hbp_set_alpha_table_g(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(arr.shape[0]), ctypes.c_int(self.num_groups))

    @staticmethod
    def clear_c(lib):
        import ctypes
        lib.hbp_set_alpha_table_g.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
        lib.hbp_set_alpha_table_g.restype = None
        lib.hbp_set_alpha_table_g(None, 0, 1)
        lib.hbp_set_damping.argtypes = [ctypes.c_double]
        lib.hbp_set_damping.restype = None
        lib.hbp_set_damping(1.0)
