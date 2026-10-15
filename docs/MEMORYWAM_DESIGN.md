# MemoryWAM implementation

## Fixed MemoryWAM

The fixed mode follows the MemoryWAM hybrid cache at every replan:

- full K/V for the first 2 episode frames (anchors);
- full K/V for the latest 4 frames;
- 8 learned gist tokens for every frame, retained for the full episode;
- visible order: anchors, all historical gist K/V in time order, recent full K/V;
- `reset_edit_memory()` clears all state at each episode boundary.

Each gist is inserted after the current reference image and before the noisy video target:

```text
text | reference image | current gist | noisy target | action
```

The stable reference/gist prefix cannot attend the noisy target. It can attend historical
memory. Future action chunks attend anchors, recent frames, and all retained gist tokens.

Exact training mode uses `batch_size=1` and all prior replans in the episode. To bound
activation memory, older history is rebuilt under `no_grad`, while the latest configured
history window (4 replans by default) and the current replan retain gradients. Older memory
is still read by the current action; it is not removed from the forward context. Samples at
different points in the episode eventually train every writer position. Current samples are
restricted to the episode-relative replan grid, so the default training sequence is exactly
`0, 16, 32, ...`, matching deployment rather than treating intermediate frames as replans.

Video flow matching still samples a random video timestep. Action conditioning and memory
writing do not reuse that noisy cache: both training and inference call the same clean-prefix
path with an empty target and video timestep zero. This keeps reference/gist K/V identical in
definition while preserving the original random-timestep video objective.

The default task config trains rank-16 FLUX.2 LoRA, the full action expert, proprio encoder,
and all memory parameters. It freezes the FLUX.2 base weights and saves trainable MoT tensors
plus explicit memory tensors, reducing optimizer/checkpoint cost without freezing the memory
writer or the policy that consumes memory.

## Adaptive history span

Enable with `memory=adaptive_span` (or `adaptive_both`). A task-conditioned controller runs
once per replan:

```math
z_t = z_{min} + (z_{max}-z_{min})\sigma(f([x_t, task, proprio]))
```

The hard span is `round(z_t)`. It gates only long-term gist entries by frame age; the two
anchors and four recent full frames remain visible. Training uses a straight-through gate
with an attention bias of `-12` for hard-excluded entries, so the forward decision is
effectively hard while task gradients can still recover a previously excluded entry.
Inference physically omits excluded gist K/V to reduce context and attention cost.

## Adaptive 1-8 gist allocation

Enable with `memory=adaptive_gist` (or `adaptive_both`). This is an adaptation of Adaptive-k,
not a claim of literal QA-retrieval reproduction. The frame/task query scores eight learned
gist candidates by cosine similarity. After sorting scores `s_(1) >= ... >= s_(8)`, the
controller forms adjacent gaps plus a no-cut option:

```math
g_k=s_{(k)}-s_{(k+1)},\quad k=1...7
```

```math
g_8=0.05,\qquad k_t=1+\arg\max_k g_k
```

Thus no clear natural break keeps all 8; a clear gap retains the top `k_t`. The soft count is
obtained from `softmax(g/tau)` and supplies the straight-through gradient. The default budget
is an upper-bound penalty, not an equality or a per-token linear tax:

```math
L_{budget}=0.001\,ReLU(E[k]-4)^2
```

The complete training loss is:

```math
L=0.5L_{video}+L_{action}+L_{budget}+L_{span}
```

The allocation for frame `t` affects memory read by later replans, so later video/action
losses backpropagate into the earlier gist writer and allocator within the retained BPTT
window. There is no cross-batch graph. The current frame's gist budget is not charged until
it has a future consumer, avoiding a one-sided pressure toward one token.

## Train/infer contract

- Controller input is the same normalized current frame, instruction embedding, and proprio.
- Controller input does not depend on diffusion timestep or future observations.
- Training and inference use the same largest-gap count and rounded history span.
- Training keeps max-size tensors with hard straight-through gates; inference compacts the
  same selected token/frame IDs. Parity tests require those IDs to match exactly.
- Training `REPLAN_STEPS` and evaluation `REPLAN_STEPS` must match (default 16).
- Checkpoints include gist, temporal/type embeddings, task projection, allocator, and span
  controller tensors in the `edit_memory` payload. A present memory payload is key/shape
  validated exactly; incompatible fixed/adaptive checkpoints fail instead of loading partly.

## Commands

```bash
GPU_PER_NODE=16 MEMORY_MODE=fixed \
ROBOTWIN_ROOT=/path/to/robotwin QWEN_CACHE_DIR=/path/to/qwen-cache \
FLUX2_SRC=/path/to/flux2 FLUX2_MODEL_PATH=/path/to/flux.safetensors \
FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors ACTION_INIT=/path/to/action_init.pt \
RESUME=/path/to/stage1_checkpoint.pt \
bash scripts/flux2/run_train_memorywam_robotwin.sh
```

Replace `MEMORY_MODE` with `adaptive_span`, `adaptive_gist`, or `adaptive_both`. Evaluation
uses `scripts/flux2/run_eval_memorywam_robotwin.sh` with the same mode and replan setting.
