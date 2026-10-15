# MemoryWAM train/infer parity contract

## L0: contract

- Train entrypoint: `scripts/flux2/run_train_memorywam_robotwin.sh`
- Inference entrypoint: `scripts/flux2/run_eval_memorywam_robotwin.sh`
- Task config: `robotwin_flux2_klein_4b_base_memorywam`
- Mode config: `fixed`, `adaptive_span`, `adaptive_gist`, or `adaptive_both`
- Checkpoint format: base `mot` plus explicit `edit_memory` tensors
- Replan interval: `REPLAN_STEPS`, default 16 in both entrypoints
- Exact checks: token/frame IDs, masks, configuration, and checkpoint tensors
- Numeric checks: BF16 hidden/action outputs use `atol=rtol=1e-3`

## L1: environment and RNG

Training uses stochastic flow-matching timesteps and noise. The memory controllers do not use
that noise: their inputs are deterministic VAE reference tokens plus task/proprio context.
Inference action noise remains controlled by the existing `seed` and `rand_device` settings.

## L2: fixed samples

`tests/test_train_infer_memory_parity.py` fixes synthetic frame IDs and controller inputs, then
reuses them for the training-selector and inference-compaction paths.

## L3: model inputs

Both paths use the same image normalization, VAE packing, Qwen3 embedding format, instruction
template, proprio projection, temporal gist embedding, and replan index. The dataset history
sampler never crosses an episode boundary, restricts current samples to the episode-relative
replan grid, and samples prior replans at the configured interval.

## L4: structure and weights

`save_checkpoint()` stores every state key prefixed by `edit_memory_`. This includes learned
gist/type/time tensors, task projection, adaptive gist allocator, and span controller. Loading
reports a warning when a Stage1 checkpoint has no memory state. A Stage2/evaluation memory
payload must have exactly the active config's keys and shapes or loading fails.
The payload also records replan interval, fixed-cache geometry, and adaptive hyperparameters;
evaluation refuses a checkpoint when any active value differs.

## L5: forward, masks, and cache

The shared selector implements `[2 anchor full K/V] + [historical gist K/V] + [4 recent full
K/V]`. Training and inference use the same count/span decisions. Training retains max-size
tensors with straight-through attention gates; inference physically compacts the exact hard
IDs. Gist tokens are explicitly ordered between reference and noisy target tokens, and the
stable prefix mask prevents future-target leakage. Action conditioning and memory writing use
one shared target-free, video-timestep-zero prefix function in both paths; random video
timestep modulation is confined to the separate video flow-matching loss.

## L6: staged verification

1. Pure cache and gradient tests.
2. Fixed training-selector versus persistent inference-bank parity.
3. Adaptive hard token/frame ID parity.
4. FLUX.2 prefix and target-slice parity.
5. Real checkpoint one-step/action-chunk smoke test before any closed-loop evaluation.
6. Closed-loop RoboTwin evaluation with one memory reset per episode.

Steps 1-4 are repository tests. Steps 5-6 require the selected checkpoint, FLUX.2 assets, text
cache, dataset stats, and simulator environment and must be recorded with their exact paths.

## L7: decoding

MemoryWAM does not change the action scheduler, denormalization, gripper conversion, action
horizon, or action execution queue. The only inference-side state changes are memory read/write
and episode reset. `memory_debug` reports the selected span/count on every replan.
