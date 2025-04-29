# **VIDAIO Subnet Validation Mechanism**

The validation mechanism in the Bittensor subnet is designed to ensure the quality and reliability of miners' contributions. Validators use two key metrics—**VMAF (Video Multi-Method Assessment Fusion)** and **PieAPP (Perceptual Image-Error Assessment through Pairwise Preference)**—to assess miners’ performance. These metrics evaluate the quality of video processing and the perceptual similarity between the original and processed videos.

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
This vmaf score is only used for thresholding, to check if the processed video is actually an upscaled version of the original file.

---

### PIE-APP (Perceptual Image-Error Assessment through Pairwise Preference)

PIE-APP measures the perceptual similarity between original and processed video frames using a deep learning-based approach.

- **Scale**: logically spans (−∞, ∞). In the positive range (0 to 5), lower values indicate better quality
- **Implementation**: The system processes frames at regular intervals (default: every frame) throughout the entire video.
- **Calculation**: `PIE-APP_score = (Σ abs(d(F_i, F'_i))) / n` where:
  - `F_i`: Frame `i` from the original video
  - `F'_i`: Corresponding frame `i` from the processed video
  - `d(F_i, F'_i)`: Perceptual difference between the two frames
  - `n`: Number of frames processed according to the frame interval
- **Calculate Final Score**: Uses mathematical formulas from average_pieapp score
  - Caps any value greater than 2.0
  - Applies sigmoid function to normalize pieapp score: `normalized_pieapp_score = 1/(1+exp(-Average_pieAPP))`
  - Calculates final score using mathematical transformation, converting from "lower is better" to "higher is better" with values normalized between 0 and 1
  - <img src="./images/graph2.png" alt="Sigmoid Function" width="900" />
  - <img src="./images/graph1.png" alt="Final Score Function" width="900" />

---

### **Combining VMAF and PieAPP Scores**

After calculating the VMAF and PieAPP scores:
- The **VMAF score** is a value between `0` and `100`.
- The **Final score** is a value between `0` and `1`.

```
The final score is based on the pieapp score, as the vmaf score serves solely as a threshold.
```

---

### **Final Scoring with Response Time**

While the **quality score** evaluates the accuracy of the video processing, the **final score** also incorporates **response time**, as speed is crucial in video processing tasks.

Validators prioritize miners who can process videos quickly without compromising quality. The **final score** is determined by combining the quality score and the miner's response time, ensuring that miners are incentivized to maintain both high-quality outputs and efficient processing speeds.

---

### **Summary of the Validation Process**
1. **VMAF Assessment**:
   - Compares every frame of the original and processed videos.
   - Uses the harmonic mean of frame scores to penalize poor-quality frames.
2. **PieAPP Assessment**:
   - Computes perceptual similarity using a deep learning-based model.
3. **Quality Score Calculation**:
   - Use PieAPP scores to produce a single quality score.
4. **Final Scoring with Response Time**:
   - Incorporates response time into the final score to prioritize speed and efficiency.

These metrics together ensure that miners maintain high-quality video processing standards while also meeting the demands for fast and efficient processing.

