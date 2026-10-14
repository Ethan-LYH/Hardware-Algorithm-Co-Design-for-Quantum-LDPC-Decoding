# Hardware-Algorithm Co-Design for Quantum LDPC Decoding

Code for the GARI-NMS decoder on bivariate-bicycle (BB) codes under circuit-level noise, plus a study of learnable normalization for it (Neural-NMS, Step 1).

- **v2**: the original reproduction code for [arXiv 2510.14060](https://arxiv.org/pdf/2510.14060) (kept unchanged; original README in [README_v2.md](README_v2.md)).
- **v4**: the same C decoder extended with per-iteration / per-row-block normalization tables and message damping, a differentiable PyTorch replica, a trainer, and diagnostic tools. All v4 files carry a `_v4` suffix, so v2 and v4 live side by side.

**Main result of the v4 study (d = 6, [[72,12,6]]):** learned or structured normalization gave no detectable logical-error-rate gain; the fixed alpha = 0.96875 without damping is already at the optimum. Details: [docs/step1_summary.md](docs/step1_summary.md).

## Repository layout

```
.
├── README.md, README_v2.md          this file / original README
├── requirements.txt, .gitignore
├── stim_batched_data_v2.py          v2 main script (paper pipeline)
├── helper_functions.py, codes_q.py, utils.py, build_circuit.py     shared code (circuits, codes, big matrix)
├── my_decoders/
│   ├── hbp_decoder_v2.c, hbplib_wrapper_v2.py      v2 decoder (original)
│   ├── Makefile                                    builds hbplib_v2.so
│   ├── hbp_decoder_v4.c, hbplib_wrapper_v4.py      v4 decoder (tables + damping)
│   └── Makefile_v4                                 builds hbplib_v4.so
├── neural_nms_v4/                   IterBeta (learnable betas) and the torch decoder replica
├── train_beta_v4.py                 train / evaluate learned betas
├── test_equivalence_v4.py           torch vs C exact comparison
├── diagnose_v4.py                   alpha sweep + failure taxonomy (C only)
├── sweep_struct_v4.py               damping and block-alpha sweeps (C only)
├── nms_problem_v4.py                torch-free problem / sampler code
├── data/
│   ├── final/                       data used for the paper plots
│   └── circuits/                    stim circuits (+ caches rebuilt automatically)
├── docs/                            research directions, Step 1 summary
├── archive/                         earlier iterations kept for history (superseded by v4, not guaranteed to run)
└── results/                         saved beta files (run logs are git-ignored)
```

Python entry points stay at the repo root on purpose: they use paths relative to the root (`my_decoders/*.so`, `data/circuits/`). **Run every command from the repo root.**

## Setup

Python 3.11 is recommended (the v2 requirement). Linux or WSL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch is only needed for train_beta_v4.py and test_equivalence_v4.py
```

CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## v2: reproduce the paper pipeline

```bash
cd my_decoders && make clean && make && cd ..
python stim_batched_data_v2.py                       # defaults: d=6, p=0.002
# explicit form (12 arguments):
python stim_batched_data_v2.py DIST P ITRS MSF RS PRIOR_TYPE CS SHOTS_K MAX_ERRS ENS POI THREADS
python stim_batched_data_v2.py 6 0.002 400 0.96875 1 0 2 16 100 0 i 8
```

`MSF` is the normalization factor alpha, `RS` the random-order seed, `CS` the custom schedule (2 = bottom, A, B), `SHOTS_K` the number of shots in units of 1024, `ENS` the ensemble size (0 or 1 = off), `POI` the ensemble selection rule (`i` = iterations, `w` = weight). Results are appended to `data/*.csv` in sinter format. The first run builds the big matrix and caches it in `data/circuits/`.

## v4: tools and experiments

Build the v4 library (does not touch `hbplib_v2.so`):

```bash
cd my_decoders && make -f Makefile_v4 && cd ..
nm -D my_decoders/hbplib_v4.so | grep hbp_set    # hbp_set_alpha_table, hbp_set_alpha_table_g, hbp_set_damping
python test_equivalence_v4.py                     # must end with ALL OK
```

All comparisons below are paired (same shots for every variant); the p-value is an exact binomial test on the shots where only one variant fails. Use a new `--seed` / `--eval_seed` to evaluate on unseen shots.

**1. Alpha sweep and failure taxonomy** (C only):

```bash
python diagnose_v4.py --d 6 --p 0.004 --shots 262144 --threads 8
python diagnose_v4.py --d 6 --p 0.002 --shots 1000000 --alphas 0.90 0.9375 0.96875 1.0 --threads 8
```

Prints LER versus scalar alpha, and splits failures into "converged to the wrong logical class" and "hit the iteration cap".

**2. Damping and block-wise alpha** (C only; blocks are A rows, B rows, tie rows):

```bash
python sweep_struct_v4.py --d 6 --p 0.004 --shots 262144 --threads 8
python sweep_struct_v4.py --d 6 --p 0.004 --shots 262144 --damp 0.9 --blocks 0.9375,0.96875,1.0
```

**3. Train per-iteration betas and evaluate** (needs torch; a GPU is used automatically if present):

```bash
python train_beta_v4.py --d 6 --p 0.004 --p_train 0.004 --steps 200 --batch 128 --train_iters 30 \
    --num_free 8 --groups 3 --loss both --hard_frac 0.5 --bank_shots 100000 --hard_iters 20 \
    --eval_shots 524288 --eval_seed 888 --eval_mitr 400 --threads 8
```

Useful options: `--groups 1|3` (one beta per iteration, or one per iteration and row block), `--loss sig|logical|both`, `--num_free N` (learn the first N betas, reuse the last), `--hard_frac` (share of hard shots per batch), `--quant_bits`. The result is saved as `beta_v4_d{d}_p{p}.json`; the script then runs the paired comparison against fixed alpha.

Evaluate a saved beta file without training:

```bash
python train_beta_v4.py --d 6 --p 0.004 --eval_beta results/beta_v4_d6_p0.004.json --eval_shots 524288 --eval_seed 888
```

**4. Use learned betas in your own decoding script** (v4 library; this snippet uses the same interface as `train_beta_v4.py`, but was not wrapped into a v4 copy of the main script):

```python
from my_decoders import hbplib_wrapper_v4 as W
from neural_nms_v4 import IterBeta

bm = IterBeta.load("results/beta_v4_d6_p0.004.json")
bm.push_to_c(W.lib)          # the C decoder now uses table[iteration]; the scalar alpha argument is ignored
# ... call W.ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler(...) as in stim_batched_data_v2.py
IterBeta.clear_c(W.lib)      # back to the scalar alpha (the table is process-global state)
```

## Notes and known limits

- The torch decoder reproduces the C decoder exactly (soft outputs, hard decisions, iteration counts) but only for the row-serial schedule; the ensemble selection logic (`ens > 1`) exists in C only.
- In the C ensemble call a shot that does not converge gets an all-zero correction, so it counts as a failure only if the true observable is flipped.
- C buffer `Hdec` is uninitialized for columns no check touches (harmless for the current matrices; use `calloc` if that changes).
- Only d = 6, one schedule seed and ensemble size 1 were studied in v4; the trained block-wise / logical-loss variants have not been evaluated yet.

## Credits

`build_circuit.py` is from Anqi Gong's [SlidingWindowDecoder](https://github.com/gongaa/SlidingWindowDecoder) (MIT license, notice kept in the file). The GARI-NMS decoder follows [arXiv 2510.14060](https://arxiv.org/pdf/2510.14060).

## 中文快速使用

1. 所有命令都在仓库根目录运行。先 `pip install -r requirements.txt`。
2. v2（论文复现）：`cd my_decoders && make && cd ..`，然后 `python stim_batched_data_v2.py`。
3. v4：`cd my_decoders && make -f Makefile_v4 && cd ..`，再运行 `python test_equivalence_v4.py`，看到 `ALL OK` 才继续。
4. 想看"α 有没有余量、失败是什么类型"：`diagnose_v4.py`；想看阻尼和分块 α：`sweep_struct_v4.py`；想训练逐迭代 β：`train_beta_v4.py`（需要 torch）。
5. v2 和 v4 文件互不覆盖，v4 文件名都带 `_v4`。
