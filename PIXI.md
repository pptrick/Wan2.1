# Pixi environment (Slurm / B200)

`pixi.toml` pins the environment for this cluster's B200 (sm_100) nodes.
`pixi.lock` is committed and is the reproducibility artifact; `.pixi/` is
generated and already covered by `.gitignore`.

## Environments

| Env       | Contents                                     | Use for                        |
|-----------|----------------------------------------------|--------------------------------|
| `default` | runtime deps + flash-attn                    | single-GPU generation          |
| `dev`     | `default` + pytest/black/flake8/isort/mypy/yapf/huggingface-hub | formatting, checkpoint download |
| `dist`    | `dev` + xfuser                               | multi-GPU (`--ulysses_size`)   |

## Important: never install or run on the login node

The login node has no GPU driver. Do everything inside an allocation:

```sh
# hold a node, then reuse it across commands
sbatch --partition=dedicated-2 --gres=gpu:8 --cpus-per-task=64 --mem=256G \
       --time=04:00:00 --wrap 'sleep 14400'
srun --jobid=<JOBID> --overlap --ntasks=1 pixi run -e dev verify
```

Raise `ulimit -d` before loading the 14B model. Some nodes set RLIMIT_DATA to
64 GiB, which counts the mmap'd checkpoint shards: diffusers gets through five
shards plus the T5 encoder (~60 GB of address space) and then fails on shard 6
with `unable to mmap ...: Cannot allocate memory` -- at only ~24 GiB resident.
Its error handler then reads the 7.9 GB shard as text and reports `MemoryError`,
which hides the real cause. The hard limit is unlimited, so:

```sh
ulimit -d unlimited
```

`debug` is often fully allocated (4 h limit, 1 node); `dedicated-2` usually has
free B200s. Check with:

```sh
scontrol show node <node> | grep -E 'AllocTRES|CfgTRES'
```

## Verify

```sh
pixi run -e dev verify
```

Inside an allocation it should print `sm_100 | kernels for it: True` and
`cudnn_sdpa usable: True`. Re-run it after any torch or flash-attn change.

## Solvers

`--sample_solver` takes `unipc` (default), `dpm++`, or `euler`.
`wan/utils/fm_solvers_euler.py` is a plain first-order baseline -- one line,
`x += (sigma_next - sigma) * v` -- on the same sigma schedule as UniPC, so a
swapped run differs only in the update rule.

Measured on 14B 720p 81f, 50 steps, seed 42, identical prompt: both solvers run
at **13.8 s/step** (the solver arithmetic is nothing against a 14B forward), but
the outputs differ substantially -- mean absolute pixel difference 15.7/255,
correlation 0.88, no identical frames. UniPC holds highlights and fine texture
noticeably better. So at Wan's defaults the corrector is buying real quality,
not just insurance.

## Attention backend

`wan/modules/attention.py` dispatches per call:

| condition                        | backend      | why                                  |
|----------------------------------|--------------|--------------------------------------|
| unpadded, no sliding window      | cuDNN SDPA   | ~2x faster than flash-attn on sm_100 |
| real padding (`min(lens) < L`)   | flash-attn   | only varlen can mask it              |
| `window_size != (-1, -1)`        | flash-attn   | SDPA cannot express it               |
| flash-attn absent                | torch SDPA   | warns; matches upstream behaviour    |

Override with `WAN_ATTN_BACKEND=auto|cudnn|flash|sdpa`. `auto` resolves per
device rather than hard-coding cuDNN -- flash-attn is generally faster on
Hopper and earlier, so this must not be pinned.

Measured on one B200, ms per sampling step (2 forwards, CFG):

| config            | cuDNN (default) | `WAN_ATTN_BACKEND=flash` | speedup |
|-------------------|-----------------|--------------------------|---------|
| 1.3B 480p 81f     | 776             | 1390                     | 1.79x   |
| 14B  480p 81f     | 3757            | 6480                     | 1.72x   |
| 14B  720p 81f     | 12729           | 26832                    | 2.11x   |

Peak memory is identical in all three. 14B at 720p peaks at 40.5 GiB, so it
fits one B200 -- multi-GPU is a throughput choice, not a memory requirement.

## Version pins that are load-bearing

- **torch `==2.9.*` (cu128).** B200 is sm_100, which needs CUDA >= 12.8; and
  flash-attn publishes no wheel past torch 2.10, ABI-tagged per torch minor.
  2.9 is the newest cu12 torch with a matching official flash-attn build.
  **Bumping torch means re-checking the flash-attn wheel URL in `pixi.toml`.**
- **flash-attn 2.8.3**, pinned by direct wheel URL. It compiles
  `arch=compute_100,code=sm_100` when CUDA >= 12.8, so it has real Blackwell
  kernels. `wan/modules/attention.py` silently falls back to torch SDPA if this
  is missing, so absence is easy to miss -- hence the explicit check in `verify`.
- **transformers `<5`, gradio `<6`.** Not in `requirements.txt`. Wan2.1 targets
  the transformers 4.x / gradio 5.x APIs; unpinned installs pulled 5.16 / 6.26.
- **huggingface-hub `<1.0`.** Forced by transformers 4.x, and it keeps the
  `[cli]` extra valid (dropped in hub 1.0) for the README's download commands.

## Gaps in the upstream dependency files

- `xfuser` is required by the multi-GPU path (`wan/distributed/xdit_context_parallel.py`
  imports it at module level) but appears in neither `requirements.txt` nor
  `pyproject.toml`. It lives in the `dist` env here.
- `yapf` is required by the Makefile's `format` target but is not in
  `pyproject.toml`'s dev extras. Added to the `dev` env.

## Known noise

`wan/modules/model.py:31,42` use the deprecated `torch.cuda.amp.autocast`
API and emit `FutureWarning` on torch 2.9. Harmless today; it would become a
hard error on a future torch.

## Upstream bugs found (not fixed here)

- `wan/modules/model.py:31,42` use the deprecated `torch.cuda.amp.autocast`
  API and emit `FutureWarning` on torch 2.9.

## Fixed here

- **`import wan` required a GPU.** `wan/modules/t5.py` had
  `device=torch.cuda.current_device()` as a default argument, which Python
  evaluates once at import time -- so importing the package raised
  `Found no NVIDIA driver` on any CPU-only machine, and otherwise froze the
  device to whichever was current at first import. Now defaults to `None` and
  resolves inside `__init__`.
- **`flash_attention()` crashed on ragged `q_lens`.** It did
  `.unflatten(0, (b, lq))` on a packed varlen result that has `sum(q_lens)`
  rows, so any ragged q raised `unflatten: Provided sizes ... don't multiply
  up`. Now scatters back into a padded `[b, lq, n, c]`. The flash-attn kernel
  itself was always correct; only this wrapper was broken. Wan never hit it
  because nothing upstream passes a ragged `q_lens`.

## Not yet verified

No Wan2.1 checkpoints are downloaded, so no end-to-end generation has been run
with real weights. Fetch one with:

```sh
pixi run -e dev huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```
