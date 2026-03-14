# Running FiberPO with verl

1. Install verl (see https://github.com/verl-project/verl).
2. From this repo root, ensure `fiberpo` is importable (e.g. `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`).
3. In your trainer entrypoint, before creating the trainer:

   ```python
   import fiberpo.losses
   fiberpo.losses.register_fiberpo_with_verl()
   ```

4. In the actor config (e.g. YAML or DictConfig), set:

   - `policy_loss.loss_mode: fiberpo`
   - `policy_loss.fiberpo_delta: 0.2`
   - `policy_loss.fiberpo_epsilon: 0.2`
   - `loss_agg_mode: seq-mean-token-mean`
   - `algorithm.adv_estimator: grpo` (or `gae`)

5. Run the trainer as usual (e.g. GRPO-style with multiple rollouts per prompt). FiberPO replaces the policy loss only; rollout and advantage computation stay the same.

For FiberPO-Domain set `loss_mode: fiberpo_domain` and provide domain/prompt_group/trajectory indices via config or batch.
