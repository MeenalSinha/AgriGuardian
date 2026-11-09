# AgriGuardian — Crop Disease Model & Explainability Architecture

**Component:** Computer Vision Intelligence Pipeline (`backend/app/services/agriculture/disease_detector.py`)  
**Target Architecture:** MobileNetV3-Small Deep Convolutional Neural Network with Squeeze-and-Excitation Attention  
**Document Version:** 1.0.0  
**Date:** 2026-11-09  

---

## 1. Architecture Selection & Agronomic Rationale

For AgriGuardian's visual disease diagnosis, **MobileNetV3-Small** was selected as the core backbone after comparing against ResNet-50 and EfficientNet-B0:

| Metric / Requirement | MobileNetV3-Small | ResNet-50 | EfficientNet-B0 | Selection Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Parameter Count** | **~2.54 Million** | ~25.6 Million | ~5.3 Million | Lightweight; minimal memory footprint |
| **Model Disk Size** | **~10.2 MB** | ~98 MB | ~21 MB | Fast download, low bandwidth consumption |
| **CPU Inference Latency** | **~28 - 45 ms** | ~180 - 240 ms | ~95 - 130 ms | Real-time response on budget hardware |
| **Top-1 Accuracy (PlantVillage)** | **98.4%** | 98.9% | 98.7% | Statistically comparable accuracy |
| **Explainability Support** | **Full Grad-CAM** | Full Grad-CAM | Full Grad-CAM | Compatible with gradient attribution |

### Architecture Highlights:
- **Depthwise Separable Convolutions:** Reduces floating-point operations by ~8-9x compared to standard 2D convolutions.
- **Squeeze-and-Excitation (SE) Residual Blocks:** Dynamically recalibrates channel-wise feature responses, allowing the network to focus on subtle lesion contours against healthy leaf tissue.
- **Hard-Swish Activation Function:** Efficient piecewise linear approximation $\text{h-swish}(x) = x \frac{\text{ReLU6}(x+3)}{6}$, eliminating expensive exponential calculations while preserving smooth gradient propagation.

---

## 2. Model Input & Output Specifications

### Input Tensor
- Shape: `[Batch_Size, 3, 224, 224]`
- Format: Normalized RGB floating-point tensor ($32$-bit float).
- Normalization: $z = \frac{x - \mu}{\sigma}$ with $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

### Output Predictions
- Output Layer: Linear classifier mapping 576-dimensional pooled features to 10 disease logits.
- Activation: Softmax $\hat{p}_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$.
- Ranking: Top-$k$ sorted probabilities with confidence score $\in [0.0, 1.0]$.

### Class Index Mapping
```json
{
  "0": "Tomato___Bacterial_spot",
  "1": "Tomato___Early_blight",
  "2": "Tomato___Late_blight",
  "3": "Tomato___Leaf_Mold",
  "4": "Tomato___Septoria_leaf_spot",
  "5": "Tomato___Spider_mites Two-spotted_spider_mite",
  "6": "Tomato___Target_Spot",
  "7": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "8": "Tomato___Tomato_mosaic_virus",
  "9": "Tomato___healthy"
}
```

---

## 3. Genuine Gradient-Weighted Class Activation Mapping (Grad-CAM)

To provide explainable AI (XAI) that farmers and agricultural officers can trust, AgriGuardian implements **genuine Grad-CAM** rather than synthetic or center-weighted heatmaps.

### Mathematical Formulation
1. **Target Convolutional Feature Maps:** Let $A^k \in \mathbb{R}^{U \times V}$ denote the activation of feature map $k$ in the final convolutional layer of MobileNetV3 (`features[-1]`).
2. **Gradient Backpropagation:** Compute the gradient of the predicted class score $y^c$ (before softmax) with respect to feature map $A^k$:
   $$\frac{\partial y^c}{\partial A^k}$$
3. **Global Average Pooling (GAP) Weights:** Calculate the importance weight $\alpha_k^c$ of feature map $k$ for class $c$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$
   where $Z = U \times V$ is the spatial area of the feature map.
4. **Weighted Linear Combination and Rectification:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ function retains features that positively contribute to the target class and filters out negative evidence.
5. **Upsampling & Blending:** The resulting $7 \times 7$ salience map is bi-linearly interpolated to the original image dimensions ($224 \times 224$ or input resolution), normalized to $[0, 1]$, colorized with a high-contrast colormap (e.g. Jet/Turbo), and blended onto the original leaf photograph with opacity $\alpha = 0.5$.

---

## 4. Evidence-Based Severity Assessment Methodology

A critical flaw in naive agricultural AI is conflating **model classification confidence** with **physical disease severity** (e.g., classifying a leaf with 1 tiny spot as "87% Severe" simply because the classifier is 87% sure it is Early Blight).

AgriGuardian **strictly decouples** model confidence from disease severity:

1. **Classifier Confidence:** Represents epistemic probability that the pathogen matches class $c$ ($P(C = c \mid \mathbf{x}) \in [0, 1]$).
2. **Physical Disease Severity:** Measures the **proportion of visible leaf surface exhibiting necrosis, chlorosis, or pest stippling**:
   $$\text{Severity \%} = \frac{\text{Area}_{\text{symptomatic}}}{\text{Area}_{\text{total leaf}}} \times 100\%$$

### Algorithmic Pipeline for Severity
1. **Leaf Surface Segmentation:**
   The image is converted to HSV color space. Green leaf tissue and chlorotic zones are segmented from background surfaces using an adaptive mask:
   $$\text{Leaf Mask} = (H \in [25, 95]) \land (S > 30) \land (V > 30)$$
2. **Lesion Segmentation:**
   Within the leaf mask, necrotic brown spots, black lesions, and severe chlorotic yellowing are segmented using color thresholding and gradient edge detection:
   $$\text{Lesion Mask} = \text{Leaf Mask} \land \left( (H \in [10, 25] \land S > 40) \lor (V < 50) \right)$$
3. **Severity Index & Clinical Staging:**
   - **Healthy:** $\text{Severity} = 0\%$ (if class is Healthy)
   - **Mild Stage:** $< 10\%$ surface area affected (early intervention window)
   - **Moderate Stage:** $10\% - 30\%$ surface area affected (active containment needed)
   - **Severe Stage:** $> 30\%$ surface area affected (critical yield loss hazard)

---

## 5. Out-of-Distribution (OOD) & Rejection Protocol

To protect farmers from incorrect pesticide or fungicide applications, the model will **abstain** when confidence is low or input quality is compromised:

- **Confidence Threshold:** $\tau = 0.40$. If $\max_i \hat{p}_i < \tau$, the system marks the prediction as `Uncertain / Unrecognized Symptom`.
- **Top-2 Entropy Margin:** If $\hat{p}_{(1)} - \hat{p}_{(2)} < 0.10$ and $\hat{p}_{(1)} < 0.60$, the system flags ambiguous multi-symptom overlap.
- **Action on Abstention:** No chemical recommendations are offered. The system displays:
  > *"Diagnostic confidence is insufficient to recommend targeted treatment. Please submit a well-lit close-up of an individual leaflet or consult the nearest Krishi Vigyan Kendra (KVK) officer."*
