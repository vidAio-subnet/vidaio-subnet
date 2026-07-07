# VIDAIO Scoring Mechanism Summary

## Overview

The VIDAIO scoring mechanism has three primary layers:

1. **Round score**: the score assigned to a miner for a specific task response.
   - Upscaling round score: `s_f`
   - Compression round score: `s_f`

2. **Long-term miner score**: the smoothed reputation score used for ranking and reward allocation.
   - Stored as `accumulate_score`

3. **Emission weighting**: the final on-chain weight calculation that allocates 80% to compression and 20% to upscaling, splits each task pool equally among its top five miners, then burns the configured proportion of miner emissions.
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

The miner manager excludes validators and miners with `accumulate_score == -1`, then separates eligible miners into compression and upscaling pools. Compression receives 80% of the pre-burn miner pool, and upscaling receives 20%.

Inside each task pool, miners are ranked by `accumulate_score` descending and the pool is split equally among the top five:

```text
rank 1           20% of that task pool
rank 2           20% of that task pool
rank 3           20% of that task pool
rank 4           20% of that task pool
rank 5           20% of that task pool
ranks 6+         0%
```

Only the top five compression miners and top five upscaling miners can receive non-zero miner-side emissions. Miners ranked 6 or lower in their task pool receive zero miner-side emission weight for that round.

After the equal top-five allocation, the miner manager applies the emissions burn:

```text
burn_proportion = 0.8

miner_weight_i = pre_burn_weight_i * (1 - burn_proportion)
burn_weight = burn_proportion * sum(pre_burn_weights)
```

With the current `burn_proportion = 0.8`, 80% of calculated miner emissions are assigned to the burn UID, which is the subnet owner UID returned by `get_burn_uid()`. The remaining 20% is distributed across the two task pools. Effective final allocations are:

```text
compression rank 1  3.2%
compression rank 2  3.2%
compression rank 3  3.2%
compression rank 4  3.2%
compression rank 5  3.2%

upscaling rank 1    0.8%
upscaling rank 2    0.8%
upscaling rank 3    0.8%
upscaling rank 4    0.8%
upscaling rank 5    0.8%
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

In summary, `s_q` represents upscaling quality, `s_f` represents the task-level final round score, `total_multiplier` adjusts the round score based on recent consistency, `accumulate_score` is the smoothed long-term score used for miner ranking, and `burn_proportion` determines how much of the calculated miner emission pool is burned before the remaining emissions reach top-ranked miners.
