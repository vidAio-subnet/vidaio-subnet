# **VIDAIO Subnet Validation Mechanism**

The validation mechanism in the Bittensor subnet is designed to ensure the quality and reliability of miners' contributions. Validators use two key metrics—**VMAF (Video Multi-Method Assessment Fusion)** and **LPIPS (Learned Perceptual Image Patch Similarity)**—to assess miners’ performance. These metrics evaluate the quality of video processing and the perceptual similarity between the original and processed videos.

---

### **VMAF (Video Multi-Method Assessment Fusion)**

VMAF is a video quality assessment metric that compares every frame of the original video with the corresponding frame in the processed (chunked) video. This metric is widely used in video processing to measure subjective video quality as perceived by humans.

#### **What VMAF Score Implies**
- A **higher VMAF score** indicates that the processed video closely resembles the original video in terms of quality.
- A **lower VMAF score** suggests significant degradation or loss of quality in the processed video.

#### **Why Use the Harmonic Mean?**
Validators use the **harmonic mean** of the VMAF scores across all frames to emphasize the impact of poor-quality frames. This approach ensures that even a single poorly processed frame has a significant effect on the overall score, thereby discouraging miners from neglecting quality in any part of the video.

The harmonic mean is particularly sensitive to low values, making it an ideal choice for scenarios where consistency in quality is critical.

#### **How VMAF is Calculated**
Let:
- `S_i`: The VMAF score for frame `i`, where `i = 1, 2, ..., n`
- `n`: Total number of frames in the video

The harmonic mean `H` of the VMAF scores is calculated as:

```
H = n / (1/S_1 + 1/S_2 + ... + 1/S_n)
```

This formula ensures that frames with lower scores have a greater impact on the overall result, penalizing poor-quality frames more heavily.

---

### **LPIPS (Learned Perceptual Image Patch Similarity)**

LPIPS is another metric used by validators to assess the perceptual similarity between the original and processed videos. Unlike VMAF, which focuses on overall video quality, LPIPS evaluates the perceptual differences between selected image patches, simulating how humans perceive visual similarity.

#### **How LPIPS is Used**
- Validators randomly select **two frames** from both the original and processed videos.
- The LPIPS score is computed by comparing these frames using a deep learning-based perceptual similarity model.

#### **What LPIPS Score Implies**
- A **lower LPIPS score** indicates higher perceptual similarity between the original and processed videos.
- A **higher LPIPS score** suggests that the processed video has noticeable perceptual differences compared to the original.

#### **Why Use Random Frame Sampling?**
Randomly sampling frames ensures that the validation process is unbiased and efficient. It reduces computational overhead while still providing a reliable assessment of perceptual similarity.

#### **How LPIPS is Calculated**
Let:
- `F_i`: Frame `i` selected randomly from the original video
- `F'_i`: Corresponding frame `i` from the processed video
- `d(F_i, F'_i)`: Perceptual difference between the two frames, as calculated by the LPIPS model

The LPIPS score `L` for the two selected frames is calculated as:

```
L = (d(F_1, F'_1) + d(F_2, F'_2)) / 2
```

This formula averages the perceptual differences across the two sampled frames, providing a robust measure of perceptual similarity.

---

### **Combining VMAF and LPIPS Scores**

After calculating the VMAF and LPIPS scores:
- The **VMAF score** is a value between `0` and `100`.
- The **LPIPS score** is a value between `0` and `1`.

These two scores are combined into a single **quality score** using the following formula:

```
quality_score = (vmaf_score / 100) * 0.6 + (1 - lpips_score) * 0.4
```

#### **Explanation of the Formula**
- The **VMAF score** contributes 60% of the total quality score, as it provides a comprehensive measure of overall video quality.
- The **LPIPS score** contributes 40% of the total quality score, emphasizing perceptual similarity between the original and processed videos.
- By combining these metrics, the formula ensures that both objective video quality and perceptual similarity are taken into account.

A higher **quality_score** indicates better performance in terms of both video quality and perceptual similarity.

---

### **Final Scoring with Response Time**

While the **quality score** evaluates the accuracy of the video processing, the **final score** also incorporates **response time**, as speed is crucial in video processing tasks.

Validators prioritize miners who can process videos quickly without compromising quality. The **final score** is determined by combining the quality score and the miner's response time, ensuring that miners are incentivized to maintain both high-quality outputs and efficient processing speeds.

---

### **Summary of the Validation Process**
1. **VMAF Assessment**:
   - Compares every frame of the original and processed videos.
   - Uses the harmonic mean of frame scores to penalize poor-quality frames.
2. **LPIPS Assessment**:
   - Randomly samples two frames from the original and processed videos.
   - Computes perceptual similarity using a deep learning-based model.
3. **Quality Score Calculation**:
   - Combines VMAF and LPIPS scores to produce a single quality score.
4. **Final Scoring with Response Time**:
   - Incorporates response time into the final score to prioritize speed and efficiency.

These metrics together ensure that miners maintain high-quality video processing standards while also meeting the demands for fast and efficient processing.

