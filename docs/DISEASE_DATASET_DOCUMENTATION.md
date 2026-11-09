# AgriGuardian — Crop Disease Dataset Documentation

**Dataset Target:** Tomato Foliar Disease Benchmark (PlantVillage & Agricultural Field Collections)  
**Task Type:** Multi-Class Image Classification & Explainable Pathogen Localization  
**Document Version:** 1.0.0  
**Date:** 2026-11-09  

---

## 1. Dataset Overview and Agronomic Scope

In Phase 2 of AgriGuardian, the visual diagnostic intelligence is focused on **Tomato (*Solanum lycopersicum*) foliar diseases**. Tomato is one of India's most critical horticultural cash crops, cultivated extensively across Andhra Pradesh, Karnataka, Madhya Pradesh, Maharashtra, Odisha, and Gujarat. 

Tomato crops are highly susceptible to severe foliar fungal, bacterial, viral, and acarological infestations that cause catastrophic yield loss (up to 80-100% in late blight or TYLCV outbreaks) if not diagnosed early.

The benchmark dataset consists of 18,160 standardized expert-annotated leaf images covering 10 distinct health states (9 pathological conditions and 1 healthy control):

| Class Index | Disease Common Name | Scientific / Causal Agent | Pathogen Type | Typical Economic Severity |
| :---: | :--- | :--- | :--- | :--- |
| `0` | **Bacterial Spot** | *Xanthomonas campestris pv. vesicatoria* | Bacteria | High (fruit & leaf lesions) |
| `1` | **Early Blight** | *Alternaria solani* | Fungus | Moderate to High (defoliation) |
| `2` | **Late Blight** | *Phytophthora infestans* | Oomycete | Critical (rapid whole-crop collapse) |
| `3` | **Leaf Mold** | *Passalora fulva* (*Cladosporium fulvum*) | Fungus | Moderate (greenhouse / high humidity) |
| `4` | **Septoria Leaf Spot** | *Septoria lycopersici* | Fungus | High (progressive bottom-up defoliation) |
| `5` | **Spider Mites** | *Tetranychus urticae* | Arachnid / Pest | Moderate (stippling, chlorosis, webbing) |
| `6` | **Target Spot** | *Corynespora cassiicola* | Fungus | Moderate to High (concentric brown lesions) |
| `7` | **Tomato Yellow Leaf Curl Virus** | *Begomovirus* (vectored by *Bemisia tabaci*) | Virus | Critical (stunting, floral abortion) |
| `8` | **Tomato Mosaic Virus** | *Tobamovirus* (ToMV) | Virus | High (leaf mottling, blistering) |
| `9` | **Healthy** | Normal leaf physiology | Non-pathogenic | None (vital crop control) |

---

## 2. Dataset Partitioning & Class Distribution

The benchmark dataset is partitioned into non-overlapping training, validation, and holdout test splits following a stratified 70% / 15% / 15% distribution:

- **Total Images:** 18,160
- **Training Set (70%):** 12,712 images
- **Validation Set (15%):** 2,724 images
- **Holdout Test Set (15%):** 2,724 images

### Distribution per Class
```
Tomato___Bacterial_spot:                    2,127 images (11.7%)
Tomato___Early_blight:                      1,000 images (5.5%)
Tomato___Late_blight:                       1,909 images (10.5%)
Tomato___Leaf_Mold:                           952 images (5.2%)
Tomato___Septoria_leaf_spot:                1,771 images (9.8%)
Tomato___Spider_mites Two-spotted:          1,676 images (9.2%)
Tomato___Target_Spot:                       1,404 images (7.7%)
Tomato___Tomato_Yellow_Leaf_Curl_Virus:     5,357 images (29.5%)
Tomato___Tomato_mosaic_virus:                 373 images (2.1%)
Tomato___healthy:                           1,591 images (8.8%)
```

---

## 3. Data Preprocessing & Augmentation Strategy

To ensure robust model generalization from laboratory-collected leaf samples to authentic Indian field conditions (variable sunlight, soil backgrounds, partial shade, and camera shake), an aggressive data augmentation pipeline is applied during training:

1. **Geometry Augmentation:**
   - Random Horizontal & Vertical Flips ($p = 0.5$)
   - Random Affine Rotation ($\pm 30^\circ$)
   - Random Resized Crop ($224 \times 224$, scale range $0.8$ to $1.0$)
2. **Photometric Augmentation:**
   - Color Jitter (Brightness $\pm 20\%$, Contrast $\pm 20\%$, Saturation $\pm 20\%$, Hue $\pm 5\%$) simulating harsh tropical sunlight and twilight conditions.
   - Gaussian Blur ($kernel=3, \sigma \in [0.1, 2.0], p = 0.2$) simulating low-cost mobile phone lens defocus.
3. **Normalization:**
   - Standard RGB z-score normalization using ImageNet statistics:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 4. Known Biases and Limitations

1. **Background Artifact Bias:**
   Many standard dataset images feature uniform laboratory backgrounds (plain paper or neutral grey boards). Models trained naively on such data can learn shortcut features (e.g., sheet edge or studio lighting). AgriGuardian addresses this through foreground leaf extraction, color jittering, and Grad-CAM spatial validation.
2. **Single-Leaf Focus:**
   The dataset captures individual leaves rather than whole-canopy photos. Users in the field are explicitly instructed via UI guidance to take close-up photos of individual affected leaflets.
3. **Co-infections & Nutritional Deficiencies:**
   The dataset assumes single-pathogen presence. Micronutrient deficiencies (e.g., magnesium or iron chlorosis) can mimic viral mottling. The system includes calibrated confidence thresholds and recommends agronomist verification when predictions are uncertain.

---

## 5. Dataset Ethics & Agronomic Safety

- **No Chemical Endorsements:** Dataset labeling is purely biological and pathogen-focused. Management outputs derived from diagnosis strictly adhere to Integrated Pest Management (IPM) guidelines approved by the Indian Council of Agricultural Research (ICAR).
- **Abstention on OOD Data:** When images of non-plant subjects, uncultivated weeds, or severely distorted leaves are submitted, the system refuses to force-fit a disease class and prompts the farmer for a clearer photo.

