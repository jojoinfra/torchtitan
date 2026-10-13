We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks
4. MoE expert-load metrics to detect routing imbalance (MoE models only)

Those metrics can then be visualized in either a TensorBoard or WandDB dashboard

## MoE expert load

For MoE models, every log step reports how evenly the routers spread tokens
across experts. The load of an expert is the number of routing assignments it
received over the whole global batch (each token counts once per selected
expert; padding is excluded). With an expert-bias update the counts are reset
every step, so the load covers that step; without one (frozen expert bias, or
`load_balance_coeff=None`) it accumulates since the start of training.
Per MoE layer:

| Statistic | Definition | Balanced value |
|---|---|---|
| `max_vio` | (max load - mean load) / mean load | 0 |
| `cv` | std / mean of the loads | 0 |
| `dead_experts` | experts that received no token | 0 |

`moe_load/max_vio_mean`, `moe_load/max_vio_max`, `moe_load/cv_mean` and
`moe_load/dead_experts` summarize them over layers (dead experts are summed).
Set `metrics.log_moe_load_per_layer` to also log
`moe_load_per_layer/{layer fqn}/{statistic}` for every MoE layer.

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a recipe function or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.

## Weights and Biases

Weights and Biases will automatically send metrics to a remote server if you login with `wandb login`

So all you need to do is make sure that `metrics.enable_wandb` is enabled

For an example you can inspect the Llama 3 [recipe](../../torchtitan_recipes/models/llama3.py)

If both W&B and TensorBoard are enabled, both loggers run.
