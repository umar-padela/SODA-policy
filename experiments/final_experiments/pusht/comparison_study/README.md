# comparison_study

Final evaluation: best SODA hierarchical policy vs Columbia DP baseline.
50 episodes each, same seeds (100000+), same max_steps=300.

## Prerequisites

- All prior studies complete; best SODA checkpoint and config identified
- DP baseline checkpoint at `/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt`

## Commands

### Eval: SODA vs DP in parallel
Replace `<soda_ckpt>` and `<soda_config>` with the best policy from termination_study.
```bash
modal run \
  experiments/final_experiments/pusht/comparison_study/modal_eval_final.py \
  --soda-checkpoint <soda_ckpt> \
  --soda-config <soda_config> \
  --dp-checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt \
  --n-action-steps <N>
```

### Plot
```bash
python experiments/final_experiments/pusht/comparison_study/plot_comparison.py \
  --data experiments/final_experiments/pusht/comparison_study/final_comparison.json
```

## Outputs

**Local:**
```
experiments/final_experiments/pusht/comparison_study/
  final_comparison.json
  plot_final_comparison.png
```

## Interpreting Results

The goal is to determine whether hierarchical decomposition (SODA) improves over flat
imitation learning (DP) on Push-T. A positive result means the structured option-conditioned
policy plus learned termination outperforms the monolithic baseline.
