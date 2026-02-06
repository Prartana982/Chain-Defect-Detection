# Model Exploration Report: Gold Chain Defect Detection

## 1. Executive Summary
We evaluated four state-of-the-art anomaly detection architectures to identify a robust solution for chain inspection.
**Selected Model: DINOv2 (ViT-S/14) + Background Masking**
- **Why?** It demonstrated the highest separation between "Good" and "Defective" samples (+0.6 margin).
- **Key finding:** Transformer-based models (DINO) focus more on the structural integrity of the chain, whereas CNN-based models (PatchCore, PaDiM) were often distracted by background texture.

## 2. Model Comparison - Visual Summary
![Separation Gap Chart](output/charts/separation_chart.png)

![F1 Score Chart](output/charts/f1_chart.png)

| Model | Backbone | Mechanism | Separation Gap (Masked) | Precision | Recall | Verdict |
|-------|----------|-----------|-------------------------|-----------|--------|---------|
| **DINOv2** | ViT-S/14 | Transformer Features + kNN | **High (+0.6)** | **1.0** | **1.0** | **🏆 Best Choice** |
| **PatchCore** | ResNet50 | CNN Features + Memory Bank | Medium (+0.4) | **1.0** | **1.0** | Good baseline, but learns background texture |
| **PaDiM** | ResNet50 | Gaussian Modeling | Medium (+0.35) | 0.8 | 0.7 | Smoother maps, but less sensitive to thin cracks |
| **CAE** | 4-Layer CNN | Reconstruction Error | Low (+0.38) | 0.8 | 0.6 | Struggles with fine details; noisy |

## 3. Detailed Visual Defect Analysis

Below are the detection results for **all defective samples** across each model. The header of each image shows the calculated **Anomaly Score**.

### A. DINOv2 (Best Performer)
<carousel>
![Broken Chain](output/DINOv2/viz_Broken-data1.png)
**Obvious Break**: Strongly detected.
<!-- slide -->
![Moderate Defect 1](output/DINOv2/viz_chain_03_defect_moderate_02.png)
**Moderate Defect**: Clear localization.
<!-- slide -->
![Moderate Defect 2](output/DINOv2/viz_chain_05_defect_moderate_01.png)
**Moderate Defect**: Good sensitivity.
<!-- slide -->
![Moderate Defect 3](output/DINOv2/viz_chain_07_defect_moderate_01.png.png)
**Moderate Defect**: Detected.
<!-- slide -->
![Subtle Defect 1](output/DINOv2/viz_chain_07_defect_subtle_01.png)
**Subtle Defect**: **Successfully detected** (Hardcase).
<!-- slide -->
![Subtle Defect 2](output/DINOv2/viz_chain_07_defect_subtle_02.png.png)
**Subtle Defect**: Detected.
</carousel>

### B. PatchCore (Runner Up)
<carousel>
![Broken Chain](output/PatchCore/viz_Broken-data1.png)
**Obvious Break**: Detected.
<!-- slide -->
![Subtle Defect 1](output/PatchCore/viz_chain_07_defect_subtle_01.png)
**Subtle Defect**: Less precise localization than DINO.
<!-- slide -->
![Subtle Defect 2](output/PatchCore/viz_chain_07_defect_subtle_02.png.png)
**Subtle Defect**: Detected.
</carousel>

### C. PaDiM
<carousel>
![Broken Chain](output/PaDiM/viz_Broken-data1.png)
**Obvious Break**: Very smooth heatmap, lacks detail.
<!-- slide -->
![Subtle Defect 1](output/PaDiM/viz_chain_07_defect_subtle_01.png)
**Subtle Defect**: Missed or very weak signal.
</carousel>

### D. CAE (Autoencoder)
<carousel>
![Broken Chain](output/CAE/viz_Broken-data1.png)
**Obvious Break**: Detected but noisy.
<!-- slide -->
![Subtle Defect 1](output/CAE/viz_chain_07_defect_subtle_01.png)
**Subtle Defect**: Significant background noise interferes with detection.
</carousel>

## 4. Final Recommendation
1.  **Architecture**: **DINOv2 (ViT-S/14)**.
2.  **Pipeline**:
    *   Input Image (1024x256)
    *   **Segmentation**: Metadata-free, contour-based masking (implemented in `src/segmentation.py`).
    *   **Feature Extraction**: DINOv2 ViT-S/14.
    *   **Scoring**: kNN distance to Normal Memory Bank.
3.  **Deployment**: Export model memory bank (~5-10MB) for fast inference.
