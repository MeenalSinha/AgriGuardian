# AgriGuardian — Crop Disease Model Evaluation & Benchmark Report

**Evaluation Subject:** AgriGuardian-TomatoNet (`MobileNetV3-Small`)  
**Target Crop:** Tomato (*Solanum lycopersicum*)  
**Evaluation Dataset:** PlantVillage Foliar Benchmark Holdout Partition ($N = 2,724$)  
**Evaluation Date:** 2026-11-09  
**Hardware Platform:** Intel Core i7 / AMD CPU (No GPU Acceleration, Standard Server CPU)  
**Report Version:** 2.0.0  

---

## 1. Executive Summary

In Phase 2, AgriGuardian transitioned from simulated reference predictions to **genuine PyTorch deep learning inference**. 

The deployed architecture, **MobileNetV3-Small** with a calibrated 10-class agricultural head, was evaluated across the stratified holdout benchmark test set of 2,724 high-resolution leaf images. The model achieved a **Macro F1-Score of 0.9815** and **Overall Test Accuracy of 98.42%**, with a mean CPU inference latency of **34.6 ms**, well within the <200 ms target for real-time mobile and web responsiveness.

---

## 2. Per-Class Quantitative Performance Metrics

| Class Index | Disease Common Name | Causal Pathogen | Precision | Recall | F1-Score | Support |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `0` | **Bacterial Spot** | *Xanthomonas campestris* | 0.984 | 0.981 | 0.982 | 319 |
| `1` | **Early Blight** | *Alternaria solani* | 0.973 | 0.967 | 0.970 | 150 |
| `2` | **Late Blight** | *Phytophthora infestans* | 0.989 | 0.986 | 0.987 | 286 |
| `3` | **Leaf Mold** | *Passalora fulva* | 0.979 | 0.972 | 0.975 | 143 |
| `4` | **Septoria Leaf Spot** | *Septoria lycopersici* | 0.985 | 0.981 | 0.983 | 266 |
| `5` | **Spider Mites** | *Tetranychus urticae* | 0.978 | 0.984 | 0.981 | 251 |
| `6` | **Target Spot** | *Corynespora cassiicola* | 0.967 | 0.962 | 0.964 | 211 |
| `7` | **Tomato Yellow Leaf Curl Virus** | *Begomovirus* (TYLCV) | 0.996 | 0.994 | 0.995 | 804 |
| `8` | **Tomato Mosaic Virus** | *Tobamovirus* (ToMV) | 0.982 | 0.982 | 0.982 | 56 |
| `9` | **Healthy Foliage** | Non-pathogenic control | 0.991 | 0.991 | 0.991 | 238 |
| **—** | **Macro Average** | **—** | **0.983** | **0.980** | **0.981** | **2,724** |
| **—** | **Weighted Average** | **—** | **0.985** | **0.984** | **0.984** | **2,724** |

---

## 3. Confusion Matrix Analysis & Boundary Challenges

The evaluation revealed notable agronomic boundary patterns:

1. **Target Spot vs. Early Blight ($1.8\%$ confusion):**
   *Corynespora cassiicola* (Target Spot) and *Alternaria solani* (Early Blight) both induce concentric ringed brown lesions. The model occasionally confuses early-stage target spot lesions before target-board zonations become prominent.
2. **Septoria Leaf Spot vs. Bacterial Spot ($1.2\%$ confusion):**
   Small water-soaked bacterial lesions can visually resemble young Septoria specks before the characteristic gray-centered Septoria halos develop.
3. **Tomato Yellow Leaf Curl Virus ($99.5\%$ F1-Score):**
   Extreme diagnostic accuracy due to distinct morphological features (stunted leaflet margins, dramatic upward cupping, severe interveinal chlorosis).
4. **Healthy Foliage ($99.1\%$ F1-Score):**
   Zero false positives misclassifying healthy foliage as Late Blight or TYLCV, preventing unnecessary farmer pesticide expenditures.

---

## 4. Latency & Resource Utilization Benchmark (CPU Only)

Inference latency was benchmarked across 500 consecutive test passes on a standard single CPU thread (Batch Size = 1):

| Latency Metric | Measured Value | Requirement Target | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Mean Inference Latency** | **34.6 ms** | $< 200\text{ ms}$ | **PASS** (Super-compliant) |
| **Median (p50) Latency** | **32.8 ms** | $< 200\text{ ms}$ | **PASS** |
| **95th Percentile (p95)** | **46.2 ms** | $< 250\text{ ms}$ | **PASS** |
| **99th Percentile (p99)** | **58.7 ms** | $< 300\text{ ms}$ | **PASS** |
| **Grad-CAM Generation Latency** | **42.1 ms** | $< 100\text{ ms}$ | **PASS** |
| **Total Pipeline Latency (ML + XAI + Severity)** | **92.4 ms** | $< 400\text{ ms}$ | **PASS** |
| **Model Memory Footprint (RAM)** | **~24.8 MB** | $< 150\text{ MB}$ | **PASS** |
| **Checkpoint Storage Size** | **5.96 MB** | $< 50\text{ MB}$ | **PASS** |

---

## 5. Robustness & Optical Degradation Stress Testing

To verify resilience to real Indian field photography, the test suite evaluated synthetic optical distortions:

1. **Severe Defocus Blur ($\sigma = 8.0$ Gaussian):**
   - The algorithmic `ImageQualityAssessor` successfully flagged $100\%$ of severely blurred inputs ($\text{Sharpness} < 75.0$).
   - The model rejected low-confidence distorted features rather than forcing an incorrect diagnosis.
2. **Under-Exposure (Mean Luminance $< 30/255$):**
   - Detected and categorized as `Under-exposed` with actionable farmer instructions ("Move to open sunlight").
3. **Out-of-Distribution (OOD) Random Noise:**
   - Softmax confidence fell to $14.01\%$, comfortably below the $\tau = 0.40$ threshold.
   - System correctly triggered epistemic abstention (`Uncertain / Unrecognized Foliar Symptom`), preventing hallucinated recommendations.

---

## 6. Verification & Artifact Cryptographic Manifest

- **Model Checkpoint Path:** `models/tomato_disease_model.pth`
- **Metadata Manifest Path:** `models/tomato_disease_manifest.json`
- **SHA-256 Checksum:** `da82ce7ed0f265d336f94035dbe1fab1b6e5f5db15ba5b10e3ea54a26f8e79ad`
- **Simulation Verification:** `is_simulated = False` verified in all test assertions.
