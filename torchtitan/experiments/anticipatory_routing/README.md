# Anticipatory Routing

Decouple MoE routing decisions from the backbone to suppress loss spikes.

At step `t` the backbone runs with the current parameters `theta_t`, but the
top-k expert *indices* are the ones computed earlier with `theta_{t-dt}`. Rather
than keeping a second copy of the model, the data for step `t` is fetched `dt`
steps early, pushed through a forward-only pass, and its routing indices cached
for later use.

A full extra forward per step is expensive, so the mode is armed dynamically:
on a detected loss spike the run rolls back to an earlier checkpoint, warms the
index cache, trains for a while with stale routing, then reverts.

## Schedule

Spike detected at step 200, rollback to step 100, `delay_steps = 11`:

| phase | prefetch forward (no grad) | train step |
|---|---|---|
| warmup x11 | data100..data110 at `theta_100` | none |
| active k=0 | data111 at `theta_100` | data100 with cached idx100 |
| active k=1 | data112 at `theta_101` | data101 with cached idx101 |
| active k | data(111+k) at `theta_(100+k)` | data(100+k) with cached idx(100+k) |
| drain x11 | none | queued data, routing computed fresh |

Every index consumed in a train step is `delay_steps` optimizer steps stale.
`AnticipatorySchedule._check_staleness` asserts that at the moment a queued
batch is taken off the queue, so a broken push/pop order fails loudly rather
than quietly reverting to fresh routing -- which trains fine and would pass any
loss-based test.

Warmup is a phase in this table but not a member of the `Phase` enum: it runs
inline inside the spike handler and takes no optimizer step, so it is a stretch
of wall-clock time rather than a state the training loop passes through. The
enum has the three states a *step* can be in.

## Usage

```bash
python -m torchtitan.train \
  --module anticipatory_routing \
  --config anticipatory_deepseek_v3_16b
```

Any recipe can be adapted with `to_anticipatory_config` in `config_registry.py`;
it installs the router override, forces CUDA graphs off, and attaches a
checkpointer. Key knobs live under `--anticipatory.*` and
`--anticipatory.detector.*`.

## Layout

| file | holds |
|---|---|
| `schedule.py` | `AnticipatorySchedule` — the phase machine, the prefetch queue, and the spike response. Everything that makes this different from ordinary training. |
| `trainer.py` | `AnticipatoryTrainer` — config validation and wiring. `train_step` runs the base step inside the schedule's step scope and does nothing else. |
| `engine.py` | The forward-only pass, mid-run rollback, and the per-step loss reduction the detector reads. |
| `router.py` | The `_select_experts` mixin and the `@override` factory that installs it. |
| `cache.py` | The index store and the scoped `capturing` / `replaying` modes routers read. |
| `detector.py` | Loss-spike detection. |

## How it works

**Router seam.** `AnticipatoryRoutingMixin` overrides
`TokenChoiceTopKRouter._select_experts`, the single producer of
`topk_expert_ids_TK`. The gating scores, routing map, per-expert token counts,
dispatcher permutation and load-balancing statistics are all derived from its
return value, so nothing else in the MoE needs to change. It also already runs
inside the `routing_decision` remat region with `recompute=False`, so it
executes exactly once per forward even under activation checkpointing.

The mixin is placed in front of whichever concrete router the config names
(`DeepSeekV3Router`, `QuantileBalancedTopKRouter`, ...) by generating the mixed
class in the `@override` factory, so `super()._select_experts` still runs that
model's own selection logic. One `override.imports` entry covers every router
subclass.

On replay the cached ids are materialized *through* `scores_TE` rather than
returned directly: the `RoutedExperts` boundary asserts a layout on
`topk_expert_ids_TK`, and a tensor read back from a plain dict carries no SPMD
type.

**Forward-only pass.** The prefetch runs under `no_grad` with the model in eval
mode. Eval mode is what keeps it from polluting training state: it is the guard
on the router's `tokens_per_expert_E` accumulation, on the MoE auxiliary loss,
and on the quantile balancer's histogram. `ntokens_seen` is not advanced, and
the decoders use no dropout, so the pass consumes no RNG.

**One data stream.** `GrainDataLoader.__iter__` returns the same stored
iterator, so a second `iter()` is an alias, not an independent stream; a second
dataloader would drift out of step. The prefetch is therefore the sole consumer
during warmup and active phases, and training consumes from an in-memory queue.
Consumption is conserved: warmup borrows `delay_steps` steps ahead and drain
repays them.

**Rollback.** `checkpointer.load(step=N)` restores model, optimizer, scheduler,
data stream and step counter. Because `load_state_dict` calls `set_state` on the
live grain iterator, the running generator rewinds in place. Non-persistent
accumulators the checkpoint does not carry (router token counts, quantile
histogram, aux-loss registers) are cleared explicitly.

**Spike detection.** The loss is smoothed with Holt's linear method -- a running
level plus a running trend -- and each step is scored against the one-step-ahead
prediction `level + trend`. The reported *onset* is the earliest step of the
contiguous run above the looser `onset_z_threshold`, because rolling back to a
checkpoint taken after the spike began would restore an already-damaged state.
Every rank is fed the same all-reduced loss, so every rank reaches the same
verdict.

Tracking the trend is not optional. Training loss declines, and a level-only
average always lags a declining series; that lag becomes the dominant term in
the residual variance and widens the band far past the real noise. Measured on a
decaying curve with noise of 0.045, a level-only detector reported sigma of 0.31
and scored a 1.9-nat spike at 3.9 sigma -- below a threshold of 6, so it never
fired at any spike amplitude. Predicting the trend leaves residuals the size of
the noise, and the same spike scores above 18.

Observations past the onset bar are held out of the update, so a sustained
excursion cannot drag the level and widen the band that is meant to catch it.
The hold is bounded by `onset_lookback`, after which the level is accepted as
the new normal.

## Validating numerics

With `--anticipatory.delay_steps 0` the schedule degenerates to "capture the
indices for batch `t`, then immediately train on batch `t` with them", so cached
indices equal fresh ones. With no dropout and eval mode changing no decoder
arithmetic, this must reproduce the baseline **bit for bit**:

```bash
python -m torchtitan.train --module deepseek_v3 --config deepseek_v3_debugmodel \
  --training.steps 10 --debug.seed 42 --debug.deterministic \
  --training.disable_cuda_graphs
```
```bash
python -m torchtitan.train --module anticipatory_routing \
  --config anticipatory_deepseek_v3_debugmodel \
  --training.steps 10 --debug.seed 42 --debug.deterministic \
  --anticipatory.delay_steps 0
```

Compare loss *and* grad_norm from the TensorBoard output with
`scripts/loss_compare.py`; stdout's five significant digits are not enough. Any
divergence means the capture pass is perturbing training state, or the injected
indices carry the wrong SPMD type.

A non-zero `delay_steps` is a computation change, so it needs convergence
evidence rather than bitwise equality.

## Limitations

- **Pipeline parallelism is rejected.** The index slot is selected once per
  `forward_backward_microbatch` call, which is exact only when a microbatch
  group holds one microbatch. A pipeline schedule interleaves the forwards of a
  group's microbatches. Supporting it needs a `routing_slot_id` threaded through
  `microbatch.model_kwargs` (which `preprocess_inputs` passes through untouched)
  and consumed by a `with_kwargs=True` forward pre-hook on each model part, plus
  `pp_schedule.eval` in the prefetch path.
- **CUDA graphs must be off.** A graph is captured once and replayed, so the
  routing mode in effect at capture time would be replayed for every later step.
  Supporting them needs fixed-address staging buffers and forced recapture on
  mode transitions.
- **No prefetch/EP-communication overlap**, so the overhead is roughly one extra
  forward per active step (+30-40%) rather than the ~20% reachable by
  overlapping the prefetch with expert-parallel all-to-all.
- `torch.compile` is supported at this seam, but only the seam has been checked.
  A minimal reproduction of `_select_experts` -- the mode branch, the capture-time
  dict write, and the replay-time dict read -- traces under `fullgraph=True` on
  torch 2.7.1 with both the eager and inductor backends, and replay returns the
  current step's indices rather than freezing the first step's into the graph.
  Mode x train/eval x grad/no-grad produced 6 unique graphs against a
  `cache_size_limit` of 8, and the schedule reaches only 3 of those combinations.
  Host transfers for `offload_indices_to_cpu` are deliberately kept out of the
  traced region -- `capture` narrows the dtype on device and `offload_slot` /
  `select` do the copies outside the forward, because a `.to("cpu")` reached
  from inside the router is captured into the graph as a per-layer,
  per-microbatch sync.
  Not yet checked: the same seam inside the real model, where it also has to
  survive the `routing_decision` remat region, the spmd_types annotations, the
  `RoutedExperts` local-SPMD boundary, activation checkpointing and FSDP. Run the
  `delay_steps=0` gate above with `--compile.components model loss` to confirm --
  a frozen-graph replay would surface there as a loss divergence from step 1.
- **Checkpointing is paused while the mode is armed.** During warmup and the
  active phase the prefetch has consumed `delay_steps` steps' worth of data, so
  the dataloader state a checkpoint would capture sits that far ahead of the
  parameters beside it; restarting from such a checkpoint would silently skip
  those batches. Saves resume once the queue drains. A consequence is that
  `always_on` writes no checkpoints at all, which is fine for the numerics gate
  above but means that recipe cannot be resumed.
- Rollback granularity is `checkpointer.interval`; reachability is bounded by
  `keep_latest_k`.
- Post-rollback steps are re-logged at step numbers that already have points, so
  the TensorBoard/W&B curve shows a fold, and checkpoints saved after a rollback
  overwrite the `step-N` directories of the abandoned trajectory.
