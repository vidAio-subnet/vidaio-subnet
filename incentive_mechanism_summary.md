# VIDAIO Scoring Mechanism Summary

## Overview

The VIDAIO scoring mechanism has three primary layers:

1. **Round score**: the score assigned to a miner for a specific task response.
   - Upscaling round score: `s_f`
   - Compression round score: `s_f`

2. **Long-term miner score**: the smoothed reputation score used for ranking and reward allocation.
   - Stored as `accumulate_score`

3. **Emission weighting**: the final on-chain weight calculation that allocates 80% to compression and 20% to upscaling, starts each task pool from equal top-five shares, optionally weighs those non-zero shares by alpha stake and recent emission liquidation behavior, then applies the configured burn proportion to miner emissions. The current burn proportion is `0.0`.
   - Implemented in `vidaio_subnet_core/validating/managing/miner_manager.py`

The scoring process evaluates both immediate task performance and recent historical consistency. A miner's reward outcome is therefore determined by whether the submitted output is valid, whether it satisfies task-specific quality requirements, whether the miner has performed consistently across recent rounds, and whether the miner ranks inside the top five for their task type.

## Upscaling Scoring

For upscaling tasks, the scorer calculates the following values:

```text
s_q   = quality score
s_l   = length score
s_pre = blended quality and length score
s_f   = final round score
```

The upscaling scoring flow is:

```text
1. Validate the submitted video:
   - file exists
   - video is valid
   - encoding is valid
   - resolution is correct
   - frame count matches expected bounds
   - file size is within the allowed limit

2. Calculate VMAF.
   If VMAF_score / 100 < VMAF_THRESHOLD:
       s_q = 0
       s_l = 0
       s_f = 0

3. If the VMAF threshold is satisfied, calculate PIEAPP.
   PIEAPP measures perceptual difference between sampled reference and output frames.

4. Convert PIEAPP into s_q.
   Lower PIEAPP values indicate better perceptual quality, so the scorer transforms
   PIEAPP into a higher-is-better quality score.

5. Calculate s_l from the requested content length:
   s_l = log(1 + content_length) / log(321)

6. Blend quality and length:
   s_pre = 0.5 * s_q + 0.5 * s_l

7. Apply the exponential final-score transformation:
   s_f = 0.1 * exp(6.979 * (s_pre - 0.5))
```

In this process, `s_q` represents output quality, `s_l` represents processed content length, and `s_f` is the final upscaling round score after blending and nonlinear amplification.

## Compression Scoring

Compression scoring does not use `s_q` or `s_l` as primary scoring dimensions. In performance history records, compression stores `s_q = 0` and `s_l = 0`; the meaningful round-level value is `s_f`.

Compression scoring begins with:

```text
C = compressed_file_size / original_file_size
R = 1 / C
```

Lower `C` indicates stronger compression. Higher `R` indicates a higher compression ratio.

The scorer first applies hard failure rules:

```text
If C >= 0.80:
    s_f = 0
```

This requires at least 1.25x compression.

```text
hard_cutoff = VMAF_threshold - 5

If VMAF_score < hard_cutoff:
    s_f = 0
```

This rejects outputs whose VMAF score is more than 5 points below the requested threshold.

If the VMAF score falls within the soft recovery zone:

```text
hard_cutoff <= VMAF_score < VMAF_threshold
```

the scorer applies:

```text
soft_zone_position = (VMAF_score - hard_cutoff) / 5
quality_factor = 0.7 * soft_zone_position^2
s_f = compression_component * quality_factor / 1.12
```

If the VMAF score meets or exceeds the requested threshold, the scorer applies:

```text
quality_component =
  0.7 + 0.3 * ((VMAF_score - VMAF_threshold) / (100 - VMAF_threshold))

s_f =
  (0.7 * compression_component + 0.3 * quality_component) / 1.12

s_f = min(1.0, s_f)
```

Compression scoring therefore primarily rewards file-size reduction, but only when quality is acceptable. Outputs with insufficient compression or severe quality loss receive a zero score.

## Historical Multipliers

After each round, the miner manager evaluates recent miner history, up to 10 rounds, and calculates consistency multipliers.

For upscaling:

```text
bonus_multiplier:
  up to +15% when recent s_f values frequently exceed 0.32

penalty_f_multiplier:
  up to -20% when recent s_f values frequently fall below 0.07

penalty_q_multiplier:
  up to -25% when recent s_q values frequently fall below 0.50

total_multiplier =
  bonus_multiplier * penalty_f_multiplier * penalty_q_multiplier
```

For compression:

```text
bonus_multiplier:
  up to +15% when recent s_f values frequently exceed 0.74

penalty_f_multiplier:
  up to -20% when recent s_f values frequently fall below 0.4

total_multiplier =
  bonus_multiplier * penalty_f_multiplier
```

Compression quality is already incorporated into `s_f` through the VMAF cutoff, soft recovery zone, and above-threshold quality component. The current implementation does not apply a separate compression quality-history multiplier.

## Accumulated Score

`accumulate_score` is the miner's long-term smoothed score. It is updated with the configured decay factor:

```text
new_accumulate_score =
  old_accumulate_score * 0.75
  + (s_f * total_multiplier) * 0.25
```

Each new round contributes 25% of the updated value, while 75% is retained from prior score history. This makes miner ranking stable while still allowing recent performance to affect rewards.

If a scorer returns `s_f = -100`, the round is skipped for accumulation. This is generally used for infrastructure or invalid-reference cases rather than ordinary miner penalties.

## Emission Allocation and Burn

Final emissions are calculated in `MinerManager.weights` in `vidaio_subnet_core/validating/managing/miner_manager.py`.

The miner manager excludes validators, identified by metagraph `validator_permit`, and miners with `accumulate_score == -1`, then separates eligible miners into compression and upscaling pools. Compression receives 80% of the pre-burn miner pool, and upscaling receives 20%.

Each miner row stores the UID's current `alpha_stake`, synced from the Bittensor metagraph `alpha_stake` vector exposed as `metagraph.alpha_stake` / `metagraph.AS`.

The miner manager also keeps a rolling `miner_emission_epoch_snapshots` table with UID, hotkey, coldkey, task type, epoch block, epoch index, owner alpha stake, stake source, metagraph emission, and timestamp. For this liquidation history, owner alpha stake means only stake from the miner's registered coldkey to its own hotkey on this subnet; stake delegated to the hotkey by any other coldkey is excluded. Rows are recorded once per `(uid, epoch_index)`, the first observation in that tempo epoch is retained, and snapshots older than the configured 10-epoch window are pruned. If the owner-pair chain query cannot be verified, no stake sample is recorded rather than recording a false zero. Legacy aggregate-hotkey snapshots are ignored.

Inside each task pool, miners are ranked by `accumulate_score` descending and the pool starts from an equal split among the top five:

```text
rank 1           20% of that task pool
rank 2           20% of that task pool
rank 3           20% of that task pool
rank 4           20% of that task pool
rank 5           20% of that task pool
ranks 6+         0%
```

Only the top five compression miners and top five upscaling miners can receive non-zero miner-side emissions. Miners ranked 6 or lower in their task pool receive zero miner-side emission weight for that round.

The optional alpha stake weighing is controlled by `CONFIG.score.alpha_stake_weigh_factor`, which defaults to `0.0`. At this default, the equal top-five split is unchanged. When the factor is positive, each task pool's non-zero recipients are multiplied by:

```text
1 + alpha_stake_weigh_factor * alpha_stake_i / sum(alpha_stake_top_nonzero_task_pool)
```

The weighted values are normalized back to the same task-pool total, so the weighing factor shifts the top-five distribution without changing the 80/20 task allocation or configured burn amount.

For example, with `burn_proportion = 0.0` and top-five alpha stakes `[150, 100, 600, 1600, 20]`, the final weights are shown below. The factor 2 and factor 5 columns illustrate the effect if alpha stake weighing is enabled:

| Rank | Alpha stake | Alpha stake share | Compression, factor 0 | Compression, factor 2 | Compression, factor 5 | Upscaling, factor 0 | Upscaling, factor 2 | Upscaling, factor 5 |
|------|-------------|-------------------|-----------------------|-----------------------|-----------------------|---------------------|---------------------|---------------------|
| 1 | 150 | 6.07% | 16.00% | 12.82% | 10.43% | 4.00% | 3.20% | 2.61% |
| 2 | 100 | 4.05% | 16.00% | 12.35% | 9.62% | 4.00% | 3.09% | 2.40% |
| 3 | 600 | 24.29% | 16.00% | 16.98% | 17.72% | 4.00% | 4.25% | 4.43% |
| 4 | 1600 | 64.78% | 16.00% | 26.23% | 33.91% | 4.00% | 6.56% | 8.48% |
| 5 | 20 | 0.81% | 16.00% | 11.61% | 8.32% | 4.00% | 2.90% | 2.08% |
| Task-pool miner total | 2470 | 100.00% | 80.00% | 80.00% | 80.00% | 20.00% | 20.00% | 20.00% |

The burn UID remains at 0% in all factor settings.

The optional emission liquidation weighing is controlled separately by `CONFIG.score.emission_liquidation_weigh_factor`, which defaults to `5.0`, and `CONFIG.score.emission_liquidation_window_epochs`, which defaults to `10` retained tempo-epoch snapshots. Epoch boundaries use `metagraph.tempo` when available, falling back to `CONFIG.SUBNET_TEMPO`. The first retained owner-coldkey snapshot is used as the alpha stake baseline, so a 10-snapshot window provides up to 9 comparable settled emission intervals. Setting the liquidation factor to `0.0` disables this layer. For each top-five non-validator miner in a task pool, the manager estimates:

```text
first_excluded_emission_i = first_snapshot.emission
total_recent_emission_i = sum(snapshot.emission over snapshots after the first boundary)
owner_alpha_stake_delta_i = max(0, last_owner_alpha_stake_i - first_owner_alpha_stake_i)
retained_emission_i = min(owner_alpha_stake_delta_i, total_recent_emission_i)
liquidated_emission_i = max(0, total_recent_emission_i - retained_emission_i)
liquidated_proportion_i = liquidated_emission_i / total_recent_emission_i
retained_proportion_i = 1 - liquidated_proportion_i
```

A positive liquidation weigh factor uses the retained side of the calculation:

```text
1 + emission_liquidation_weigh_factor * retained_proportion_i
```

The first boundary emission is excluded because the rolling window does not include the previous owner alpha stake baseline needed to compare it. Emissions after the first boundary are treated as settled interval emissions and compared with the miner-owner coldkey's stake change across the same window. Changes made by outside coldkeys therefore cannot alter this signal. The values are normalized back to the same task-pool total. Miners with fewer than two verified owner-coldkey snapshots or no comparable recent emissions are assumed to have liquidated 50% of recent emissions, so their fallback retained signal is `0.5`. If the whole top-five task pool is missing history, every miner receives the same fallback signal and this layer leaves the split unchanged after normalization.

For example, with `burn_proportion = 0.0`, no alpha stake weighing, and top-five recent liquidation percentages `[40%, 20%, 70%, 10%, 100%]`, the final weights are:

| Rank | Liquidated | Retained signal | Compression, factor 0 | Compression, factor 2 | Compression, factor 5 | Upscaling, factor 0 | Upscaling, factor 2 | Upscaling, factor 5 |
|------|------------|-----------------|-----------------------|-----------------------|-----------------------|---------------------|---------------------|---------------------|
| 1 | 40% | 60% | 16.00% | 17.25% | 17.78% | 4.00% | 4.31% | 4.44% |
| 2 | 20% | 80% | 16.00% | 20.39% | 22.22% | 4.00% | 5.10% | 5.56% |
| 3 | 70% | 30% | 16.00% | 12.55% | 11.11% | 4.00% | 3.14% | 2.78% |
| 4 | 10% | 90% | 16.00% | 21.96% | 24.44% | 4.00% | 5.49% | 6.11% |
| 5 | 100% | 0% | 16.00% | 7.84% | 4.44% | 4.00% | 1.96% | 1.11% |
| Task-pool miner total | n/a | n/a | 80.00% | 80.00% | 80.00% | 20.00% | 20.00% | 20.00% |

When both `alpha_stake_weigh_factor = 5.0` and `emission_liquidation_weigh_factor = 5.0`, using top-five alpha stakes `[150, 100, 600, 1600, 20]` and recent liquidation percentages `[40%, 20%, 70%, 10%, 100%]`, the final weights are:

| Rank | Alpha stake | Liquidated | Alpha-only factor 5, compression | Both factors 5, compression | Alpha-only factor 5, upscaling | Both factors 5, upscaling |
|------|-------------|------------|----------------------------------|-----------------------------|--------------------------------|---------------------------|
| 1 | 150 | 40% | 10.43% | 10.15% | 2.61% | 2.54% |
| 2 | 100 | 20% | 9.62% | 11.70% | 2.40% | 2.92% |
| 3 | 600 | 70% | 17.72% | 10.77% | 4.43% | 2.69% |
| 4 | 1600 | 10% | 33.91% | 45.36% | 8.48% | 11.34% |
| 5 | 20 | 100% | 8.32% | 2.02% | 2.08% | 0.51% |
| Task-pool miner total | 2470 | n/a | 80.00% | 80.00% | 20.00% | 20.00% |

After the base top-five allocation, optional alpha stake weighing, and optional emission liquidation weighing, the miner manager applies the emissions burn:

```text
burn_proportion = 0.0

pre_burn_weight_i = weighted top-five allocation for miner i
miner_weight_i = pre_burn_weight_i * (1 - burn_proportion)
burn_weight = burn_proportion * sum(pre_burn_weights)
```

With `burn_proportion = 0.0`, `alpha_stake_weigh_factor = 0.0`, and `emission_liquidation_weigh_factor = 0.0`, no calculated miner emissions are assigned to the burn UID, which is the subnet owner UID returned by `get_burn_uid()`. The full calculated miner pool is distributed across the two task pools. Effective final allocations are:

```text
compression rank 1  16%
compression rank 2  16%
compression rank 3  16%
compression rank 4  16%
compression rank 5  16%

upscaling rank 1    4%
upscaling rank 2    4%
upscaling rank 3    4%
upscaling rank 4    4%
upscaling rank 5    4%
```

## Performance Tier

`performance_tier` is a label derived from the miner's average recent `s_f`:

```text
> 0.40  Elite
> 0.30  Outstanding
> 0.25  High Performance
> 0.20  Good Performance
> 0.10  Average
> 0.07  Below Average
else    Poor Performance
```

In summary, `s_q` represents upscaling quality, `s_f` represents the task-level final round score, `total_multiplier` adjusts the round score based on recent consistency, `accumulate_score` is the smoothed long-term score used for miner ranking, `alpha_stake_weigh_factor` optionally weighs non-zero top-five task-pool allocations by alpha stake, `emission_liquidation_weigh_factor` optionally weighs those allocations toward miners that retain more recent emissions, and `burn_proportion` determines how much of the calculated miner emission pool is burned before the remaining emissions reach top-ranked miners. The current `burn_proportion` is `0.0`, so no miner emissions are burned by default.
