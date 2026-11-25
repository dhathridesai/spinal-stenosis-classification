# Spinal Stenosis Severity Classification (MRI)

Deep learning model that classifies lumbar spinal stenosis severity from MRI images into four classes: **normal, mild, moderate, severe**.

This repository contains the **ResNet50 transfer-learning baseline** from my research on automated spinal stenosis detection, published at ICARECS 2025 (Atlantis Press): [DOI 10.2991/978-94-6463-754-0_47](https://doi.org/10.2991/978-94-6463-754-0_47).

## Approach
- **Model:** ResNet50 pretrained on ImageNet (frozen) + custom classification head (Dense 256 → 64 → softmax, BatchNorm, Dropout 0.3)
- **Input:** 224×224 RGB MRI slices, horizontal-flip augmentation
- **Data split:** 1,983 images → 80% train / 10% validation / 10% test
- **Training:** Adam optimizer, categorical cross-entropy, 20 epochs
- **Tools:** Python, TensorFlow/Keras, scikit-learn, pandas, Matplotlib

## Results (test set, 198 images)

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Mild     | 0.40      | 0.34   | 0.37 |
| Moderate | 0.80      | 0.91   | 0.85 |
| Normal   | 0.74      | 0.81   | 0.77 |
| Severe   | 0.32      | 0.20   | 0.24 |
| **Overall accuracy** | | | **0.67** |

## Observations
- The model **overfits**: training accuracy reached 93% while validation accuracy plateaued around 59%.
- Moderate and normal cases are classified well; **mild and severe are often confused**, likely due to class imbalance and differences between image sources.

## Next steps (in progress)
- Fine-tune upper layers with **ResNet50V2**
- Stronger augmentation (rotation, zoom, brightness), class weighting, early stopping
- Save the best checkpoint on validation accuracy, not training accuracy
- **Grad-CAM** heatmaps to visualize which regions drive predictions
- Deploy as a Streamlit web app

## Dataset
Lumbar spine MRI images sourced from Kaggle (not included in this repo). *Link: [add dataset link here]*

## Publication
P. C. Reddy, **Y. D. Desai**, M. Sudamani. *Automated Detection and Classification of Spinal Stenosis for Enhanced Accuracy.* ICARECS 2025, Atlantis Press.
