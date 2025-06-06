# VIDAIO Subnet Validation & Incentive Mechanism

## Table of Contents
- [Overview](#overview)
- [Validation Metrics](#validation-metrics)
  - [VMAF (Video Multi-Method Assessment Fusion)](#vmaf-video-multi-method-assessment-fusion)
  - [PIE-APP (Perceptual Image-Error Assessment)](#pie-app-perceptual-image-error-assessment-through-pairwise-preference)
  - [Score Integration](#combining-vmaf-and-pie-app-scores)
- [Scoring System](#scoring-system)
  - [Content Length Scoring](#content-length-score-calculation)
  - [Final Score Calculation](#final-score-calculation)
  - [Penalty & Bonus System](#penalty--bonus-system)
- [Implementation Guidelines](#implementation-notes)
- [Technical Specifications](#technical-specifications)
- [Mathematical Properties](#mathematical-properties)

---

## Overview

The VIDAIO subnet validation mechanism ensures quality and reliability of miners' contributions through a comprehensive dual-metric assessment system. This mechanism evaluates video processing performance using **VMAF** and **PIE-APP** metrics, combined with an advanced scoring system that rewards consistency and penalizes poor performance.

**Key Features:**
- 🎯 **Dual-metric validation** using industry-standard VMAF and PIE-APP
- 📊 **Dynamic content length processing** (5s to 320s capability)
- 🏆 **Performance-based incentive system** with exponential rewards
- 📈 **Historical performance tracking** with rolling 10-round windows
- ⚖️ **Balanced penalty/bonus multipliers** encouraging sustained excellence

---

## Upscaling Quality Validation Metrics

### VMAF (Video Multi-Method Assessment Fusion)

VMAF serves as the foundational video quality assessment metric, comparing frame-by-frame quality between original and processed videos. This metric provides objective measurement of subjective video quality as perceived by humans.

#### Key Characteristics
- **Purpose**: Frame-by-frame quality comparison
- **Range**: 0-100 (higher values indicate better quality)
- **Usage**: Threshold validation for upscaling verification
- **Industry Standard**: Widely adopted in professional video processing

#### Mathematical Implementation

**Harmonic Mean Calculation:**
```
H = n / (1/S_1 + 1/S_2 + ... + 1/S_n)
```

**Where:**
- `S_i`: VMAF score for frame `i` (i = 1, 2, ..., n)
- `n`: Total number of frames in the video
- `H`: Harmonic mean emphasizing poor-quality frame impact

#### Why Harmonic Mean?

The harmonic mean approach provides several critical advantages:

| Advantage | Description | Impact |
|-----------|-------------|---------|
| **Sensitivity to Low Values** | Heavily penalizes poor-quality frames | Ensures consistent quality |
| **Quality Consistency** | Prevents miners from neglecting frame quality | Maintains processing standards |
| **Threshold Function** | Validates authentic upscaling processes | Prevents gaming attempts |

> **Note**: VMAF scores are used exclusively for threshold validation to verify that processed videos are genuine upscaled versions of original content. We take 5 random frames for vmaf score calculation

---

### PIE-APP (Perceptual Image-Error Assessment through Pairwise Preference)

PIE-APP provides deep learning-based perceptual similarity assessment between original and processed video frames, serving as the primary quality scoring mechanism.

#### Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Scale Range** | (-∞, ∞) | Theoretical range |
| **Practical Range** | 0 to 5+ | Positive values (lower = better) |
| **Processing Interval** | Every frame | Default frame sampling rate |
| **Implementation** | Deep learning-based | Advanced perceptual assessment |

#### Calculation Process

**Step 1: Raw PIE-APP Score**
```
PIE-APP_score = (Σ abs(d(F_i, F'_i))) / n
```

**Where:**
- `F_i`: Frame `i` from original video
- `F'_i`: Corresponding frame `i` from processed video  
- `d(F_i, F'_i)`: Perceptual difference between frames
- `n`: Number of processed frames

**Step 2: Score Normalization**
```
1. Cap values: max(Average_PIE-APP, 2.0)
2. Sigmoid normalization: normalized_score = 1/(1+exp(-Average_PIE-APP))
3. Final transformation: Convert "lower is better" to "higher is better" (0-1 range)
4. We take 4 random frames for comparison
```

#### Visual Score Transformation

The PIE-APP scoring system uses sophisticated mathematical transformations:

- *Sigmoid normalization function for PIE-APP scores*
!<img src="./images/graph2.png" alt="Sigmoid Function" width="1200" height="800">

- *Final score transformation converting to 0-1 range*
!<img src="./images/graph1.png" alt="Final Score Function" width="1200" height="800">

---

### Combining VMAF and PIE-APP Scores

The validation system integrates both metrics with distinct roles:

| Metric | Range | Primary Function | Usage |
|--------|-------|------------------|-------|
| **VMAF** | 0-100 | Threshold validation | Upscaling verification |
| **PIE-APP** | 0-1 (final) | Quality scoring | Performance evaluation |

```
Final Quality Score S_Q = PIE-APP Final Score
(VMAF serves as threshold validation only)
```

---

## Combination between Quality score and Content Length score

### Content Length Score Calculation

#### Dynamic Content Length Requests

Miners actively request content processing durations within 60-second evaluation windows, enabling optimized resource allocation and performance assessment.

#### Available Processing Durations

| Duration | Status | Availability |
|----------|--------|--------------|
| 5s | ✅ Default | Currently Available |
| 10s | ✅ Available | Currently Available |
| 20s | ✅ Available | Currently Available |
| 40s | 🔄 Coming Soon | Future Release |
| 80s | 🔄 Coming Soon | Future Release |
| 160s | 🔄 Coming Soon | Future Release |
| 320s | 🔄 Coming Soon | Future Release |


> **Current Limitation**: Processing durations up to 20 seconds are currently supported.


#### Length Score Mathematical Model

**Formula:**
```
S_L = log(1 + content_length) / log(1 + 320)
```

**Parameters:**
- `content_length`: Processing duration in seconds
- `S_L`: Normalized length score (0 to 1)

#### Performance Analysis Table

| Duration (s) | S_L Score | Percentage | Improvement | Performance Tier |
|--------------|-----------|------------|-------------|------------------|
| 5 | 0.3105 | 31.05% | Baseline | Default |
| 10 | 0.4155 | 41.55% | +33% | Significant Gain |
| 20 | 0.5275 | 52.75% | +27% | Strong Performance |
| 40 | 0.6434 | 64.34% | +22% | High Capability |
| 80 | 0.7614 | 76.14% | +18% | Advanced Processing |
| 160 | 0.8804 | 88.04% | +16% | Expert Level |
| 320 | 1.0000 | 100.00% | +14% | Maximum Score |


#### Logarithmic Scaling Benefits

| Benefit | Description | Impact |
|---------|-------------|---------|
| **Fair Distribution** | Balanced scoring across duration ranges | Equitable competition |
| **Diminishing Returns** | Reduced gains for extreme durations | Prevents over-optimization |
| **Normalized Output** | Consistent 0-1 scoring range | Standardized evaluation |
| **Capacity Recognition** | Rewards longer processing capabilities | Incentivizes advancement |

**Strategic Insights:**
- **Optimal Entry Point**: 10s processing provides largest relative improvement (+33%)
- **Scaling Pattern**: Each duration doubling yields progressively smaller benefits
- **Maximum Achievement**: 320s processing represents theoretical performance ceiling

---

### Final Score Calculation

#### Score Component Architecture

The comprehensive scoring system integrates two fundamental metrics:

| Component | Symbol | Description | Weight |
|-----------|--------|-------------|--------|
| **Quality Score** | S_Q | Processing accuracy and output quality | W1 = 0.5 |
| **Length Score** | S_L | Content processing capacity | W2 = 0.5 |

#### Preliminary Score Calculation

**Formula:**
```
S_pre = S_Q × W1 + S_L × W2
```

**Current Configuration:**
- `W1 = 0.5` (Quality weight)
- `W2 = 0.5` (Length weight)

> **Dynamic Adjustment**: Weights are continuously optimized based on real-world performance data and network requirements.

#### Final Score Transformation

The preliminary score undergoes exponential transformation for enhanced performance differentiation:

**Formula:**
```
S_F = 0.1 × e^(6.979 × (S_pre - 0.5))
```

**Parameters:**
- `S_F`: Final score
- `S_pre`: Preliminary combined score
- `e`: Euler's number (≈2.718)

#### Performance Tier Analysis

| S_pre | S_F Score | Multiplier | Performance Tier | Reward Category |
|-------|-----------|------------|------------------|-----------------|
| 0.30 | 0.0248 | 0.25× | Poor Performance | Significant Penalty |
| 0.36 | 0.0376 | 0.38× | Below Average | Moderate Penalty |
| 0.42 | 0.0572 | 0.57× | Low Average | Minor Penalty |
| 0.48 | 0.0870 | 0.87× | Near Average | Slight Penalty |
| 0.54 | 0.1322 | 1.32× | Above Average | Moderate Reward |
| 0.60 | 0.2010 | 2.01× | Good Performance | Strong Reward |
| 0.66 | 0.3055 | 3.05× | High Performance | Major Reward |
| 0.72 | 0.4643 | 4.64× | Very High Performance | Excellent Reward |
| 0.78 | 0.7058 | 7.06× | Excellent Performance | Outstanding Reward |
| 0.84 | 1.0728 | 10.73× | Outstanding Performance | Elite Reward |
| 0.90 | 1.6307 | 16.31× | Elite Performance | Maximum Reward |

#### Exponential Function Characteristics

**System Benefits:**
| Feature | Description | Strategic Impact |
|---------|-------------|------------------|
| **Enhanced Differentiation** | Clear performance tier separation | Competitive advantage clarity |
| **Reward Amplification** | 16× multiplier difference (top vs bottom) | Strong performance incentives |
| **Competitive Optimization** | Non-linear improvement rewards | Encourages continuous advancement |
| **Exponential Scaling** | Small S_pre gains yield large S_F improvements | High-performance focus |

**Strategic Performance Guidelines:**
- **Minimum Target**: Achieve S_pre > 0.6 for meaningful reward activation
- **Optimization Focus**: Exponential curve creates powerful excellence incentives
- **High-Performance Strategy**: Small quality improvements at elevated levels yield disproportionate benefits

#### Graph analysis

!<img src="./images/graph3.png" alt="Length Score Analysis" width="1200" height="1000">

---

### Penalty & Bonus System

#### Historical Performance Multiplier Architecture

The advanced scoring system incorporates a **rolling 10-round historical performance window** to evaluate consistency patterns and apply dynamic multipliers based on sustained performance trends.

#### System Formula

```
Final Adjusted Score = S_F × Performance Multiplier
Performance Multiplier = Bonus Multiplier × S_F Penalty × S_Q Penalty
```

---

#### Bonus System (Excellence Rewards)

**Activation Criteria:** `S_F > 0.77` in mining round

**Mathematical Model:**
```python
bonus_multiplier = 1.0 + (bonus_count / 10) × 0.15
```

**System Characteristics:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Maximum Bonus** | +15% | All 10 rounds achieve S_F > 0.77 |
| **Scaling Method** | Linear | Based on consistency frequency |
| **Primary Purpose** | Sustained excellence reward | Long-term performance incentive |

**Example Calculation:** 7/10 rounds with S_F > 0.77 → 1.105× multiplier (+10.5% bonus)

---

#### S_F Penalty System (Performance Penalties)

**Activation Criteria:** `S_F < 0.45` in mining round

**Mathematical Model:**
```python
penalty_f_multiplier = 1.0 - (penalty_f_count / 10) × 0.20
```

**System Characteristics:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Maximum Penalty** | -20% | All 10 rounds achieve S_F < 0.45 |
| **Scaling Method** | Linear | Based on poor performance frequency |
| **Primary Purpose** | Performance consistency enforcement | Discourages sustained poor results |

**Example Calculation:** 4/10 rounds with S_F < 0.45 → 0.92× multiplier (-8% penalty)

---

#### S_Q Penalty System (Quality Penalties)

**Activation Criteria:** `S_Q < 0.5` in mining round

**Mathematical Model:**
```python
penalty_q_multiplier = 1.0 - (penalty_q_count / 10) × 0.25
```

**System Characteristics:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Maximum Penalty** | -25% | All 10 rounds achieve S_Q < 0.5 |
| **Scaling Method** | Linear | Based on quality failure frequency |
| **Primary Purpose** | Quality standard enforcement | Strongest penalty (quality is critical) |

**Example Calculation:** 3/10 rounds with S_Q < 0.5 → 0.925× multiplier (-7.5% penalty)

---

#### Performance Multiplier Case Studies

| Miner Category | Avg S_F | Avg S_Q | Bonus Rate | S_F Penalty | S_Q Penalty | **Final Multiplier** | **Net Effect** |
|----------------|---------|---------|------------|-------------|-------------|---------------------|----------------|
| **Elite Miner** | 0.854 | 0.717 | 10/10 | 0/10 | 0/10 | **1.150×** | **+15.0%** |
| **Good Miner** | 0.653 | 0.571 | 0/10 | 0/10 | 0/10 | **1.000×** | **±0.0%** |
| **Average Miner** | 0.511 | 0.505 | 0/10 | 0/10 | 3/10 | **0.925×** | **-7.5%** |
| **Poor Miner** | 0.311 | 0.411 | 0/10 | 10/10 | 10/10 | **0.600×** | **-40.0%** |

#### Penalty Analysis

!<img src="./images/graph4.png" alt="Length Score Analysis" width="1200" height="800">


---

#### System Benefits & Strategic Impact

**Core System Benefits:**

| Benefit | Description | Strategic Impact |
|---------|-------------|------------------|
| 🎯 **Consistency Rewards** | Elite miners maintain sustained +15% bonus | Long-term competitive advantage |
| ⚡ **Responsive Penalties** | Poor performance accumulates immediate penalties | Rapid feedback mechanism |
| 🔄 **Recovery Incentive** | Miners can improve multipliers over 10 rounds | Encourages continuous improvement |
| ⚖️ **Balanced Impact** | Quality penalties are strongest (-25% max) | Emphasizes quality importance |
| 📈 **Progressive Scaling** | Linear scaling prevents extreme swings | Maintains system stability |

---

## Implementation Notes

### Performance Monitoring

**Real-time Operations:**
- ✅ Scores calculated in real-time during mining operations
- ✅ Historical performance data maintained for trend analysis and multiplier calculation
- ✅ Weight adjustments implemented based on network-wide performance metrics
- ✅ Performance multipliers updated after each mining round

### Scoring Strategy Recommendations

**Performance-Based Guidelines:**

| Miner Category | Primary Focus | Strategic Recommendations |
|----------------|---------------|---------------------------|
| **New Miners** | Foundation Building | Focus on achieving consistent S_pre > 0.5 before optimizing for length |
| **Established Miners** | Quality Optimization | Prioritize quality improvements when S_pre > 0.6 to avoid S_Q penalties |
| **Elite Miners** | Consistency Maintenance | Maintain consistency above S_F > 0.77 to secure maximum bonus multipliers |
| **Recovery Phase** | Systematic Improvement | Focus on quality (S_Q > 0.5) first, then performance (S_F > 0.45) to restore multipliers |

### Future Enhancement Roadmap

**Planned Developments:**

- [ ] **Extended Content Length Support** - Processing durations up to 320s
- [ ] **Dynamic Weight Adjustment Algorithms** - Automated optimization based on network performance
- [ ] **Advanced Quality Metrics Integration** - Additional assessment parameters
- [ ] **Multi-dimensional Scoring Parameters** - Enhanced evaluation criteria
- [ ] **Adaptive Difficulty Scaling** - Network performance-based adjustments
- [ ] **Advanced Penalty/Bonus Optimization** - Network-wide performance distribution analysis
- [ ] **Seasonal Performance Multiplier Adjustments** - Time-based optimization cycles

---

## Technical Specifications

### Core System Parameters

| Parameter | Current Value | Configurable Range | Implementation Notes |
|-----------|---------------|-------------------|---------------------|
| **Default Content Length** | 5s | 5s - 40s | Actively configurable by miners |
| **Quality Weight (W1)** | 0.5 | 0.0 - 1.0 | Dynamically adjusted based on network data |
| **Length Weight (W2)** | 0.5 | 0.0 - 1.0 | Dynamically adjusted based on network data |
| **Exponential Coefficient** | 6.979 | Fixed | Dynamically adjusted based on network data |
| **Score Transformation Base** | 0.1 | Fixed | Baseline multiplier constant |

### Performance Multiplier System Parameters

| Parameter | Current Value | Configurable Range | System Impact |
|-----------|---------------|-------------------|---------------|
| **Performance History Window** | 10 rounds | 5-20 rounds | Configurable for different network conditions |
| **Bonus Threshold** | S_F > 0.77 | 0.7-0.85 | Adjustable based on network performance |
| **S_F Penalty Threshold** | S_F < 0.45 | 0.3-0.5 | Adjustable based on network performance |
| **S_Q Penalty Threshold** | S_Q < 0.5 | 0.4-0.6 | Adjustable based on network performance |
| **Maximum Bonus** | +15% | 10%-20% | Scalable reward system |
| **Maximum S_F Penalty** | -20% | 15%-25% | Scalable penalty system |
| **Maximum S_Q Penalty** | -25% | 20%-30% | Strongest penalty for quality issues |

---

## Mathematical Properties

### Length Score Function Properties

**Mathematical Characteristics:**
- **Domain**: [5, 320] seconds
- **Range**: [0.3105, 1.0000]
- **Function Type**: Logarithmic (concave)
- **Growth Rate**: Decreasing marginal returns
- **Optimization Point**: Balanced between processing capability and diminishing returns

### Final Score Function Properties

**Mathematical Characteristics:**
- **Domain**: [0, 1] (S_pre values)
- **Range**: [0.0025, 40.43] (theoretical maximum)
- **Function Type**: Exponential (convex)
- **Critical Point**: S_pre = 0.5 (inflection point for reward/penalty)
- **Scaling Behavior**: Exponential amplification of performance differences

### Performance Multiplier Properties

**Mathematical Characteristics:**
- **Domain**: [0.55, 1.15] (practical operational range)
- **Function Type**: Linear combination of historical performance frequencies
- **Update Frequency**: After each mining round completion
- **Memory System**: Rolling 10-round window with automatic history management
- **Convergence**: Stabilizes after 10 rounds of consistent performance patterns

---

## Conclusion

The VIDAIO subnet validation and incentive mechanism represents a comprehensive, mathematically-grounded approach to ensuring high-quality video processing while maintaining fair competition and encouraging continuous improvement. Through the integration of industry-standard metrics (VMAF and PIE-APP), dynamic scoring systems, and sophisticated penalty/bonus mechanisms, the system creates a robust environment that rewards excellence and consistency while providing clear pathways for improvement.

These metrics together ensure that miners maintain high-quality video processing standards while meeting demands for fast and efficient processing, creating a sustainable and competitive ecosystem for video enhancement services.

---

*This documentation is continuously updated to reflect the latest scoring mechanisms, performance optimizations, and system enhancements.*