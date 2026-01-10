# Experiment 3: Augmentation Comparison - Baseline vs Field-Style

This experiment compares two models trained with identical architecture and hyperparameters, but with different augmentation strategies, to evaluate the impact of field-style augmentation on generalization to real-world field images.

## Objective

Quantify whether field-style augmentation improves generalization to real-world field images **without exposure to field data during training**.

## Experimental Design

### Models

1. **Baseline Augmentation Model**
   - Standard preprocessing
   - Resize / center crop
   - Optional horizontal flip
   - Normalization only
   - No field-style distortions

2. **Field-Style Augmentation Model**
   - Includes augmentations designed to simulate real-world acquisition conditions:
     - Strong color jitter (illumination variability)
     - Random rotations and scale changes
     - Blur (motion / defocus)
     - Background variability / random crops
     - Mild occlusions
   - Same normalization as baseline

### Key Requirements

- Both models are **identical** in architecture (EfficientNet-B0) and hyperparameters
- Both trained on **PlantVillage dataset only** (no field data during training)
- Same normalization for both models
- Evaluation on PlantVillage test set (in-domain) and Field dataset (out-of-domain)
- Field dataset evaluation: Only intersection classes between PlantVillage and Field dataset
- **No retraining, fine-tuning, or adaptation** on field data

## Training Configuration

- **Model Architecture**: EfficientNet-B0 (trained from scratch)
- **Learning Rate**: 3e-4
- **Weight Decay**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Max Epochs**: 20
- **Early Stopping Patience**: 5
- **Batch Size**: 32
- **Loss Function**: CrossEntropyLoss

## Evaluation Protocol

### In-Domain Evaluation (Sanity Check)

- **Test Set**: PlantVillage test split
- **Classes**: All PlantVillage classes (39 classes)
- **Purpose**: 
  - Measure in-domain performance
  - Quantify any performance trade-off caused by stronger augmentation
- **Metrics**: Macro-averaged F1 score (or accuracy for comparison)

### Out-of-Domain Evaluation (Generalization Test)

- **Test Set**: Real-world field dataset
- **Classes**: Intersection of PlantVillage and field dataset classes only
- **No retraining, fine-tuning, or adaptation** is performed
- **Metrics**:
  - Macro-averaged F1 score
  - Per-class F1 scores
  - Confusion matrix

## Files

- `1_train_augmentation_comparison.ipynb`: Main training and evaluation notebook
- `experiment_3_summary.txt`: Summary of results (generated after running the notebook)
- `confusion_matrices_field_dataset.png`: Confusion matrices for field dataset evaluation (generated after running the notebook)

## Expected Outcome Interpretation

### If Field-Style Augmentation Improves Performance:

- **Supports the hypothesis** that domain-invariant features can be learned from clean data alone
- Indicates that strong augmentation can simulate real-world conditions effectively

### If Field-Style Augmentation Does Not Improve Performance:

- **Suggests the need for**:
  - Robust pretraining
  - Domain adaptation techniques
  - Unlabeled real-world data
- Indicates that augmentation alone may not be sufficient for domain generalization

### Analysis Goals

1. Quantify the performance gap between baseline augmentation vs field-style augmentation on real-world data
2. Determine whether field-style augmentation:
   - Improves generalization without exposure to real-world images
   - Introduces acceptable degradation on in-domain performance
3. Identify disease classes that remain challenging under domain shift

## Running the Experiment

1. Open `1_train_augmentation_comparison.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook will:
   - Train both models (this may take several hours)
   - Evaluate on PlantVillage test set
   - Evaluate on Field dataset (intersection classes only)
   - Generate per-class F1 scores
   - Generate confusion matrices
   - Create a summary report

## Results

Results will be saved to:
- `experiment_3_summary.txt`: Text summary of all metrics
- `confusion_matrices_field_dataset.png`: Visual confusion matrices
- Model checkpoints saved to `../models/`:
  - `efficientnet_b0_baseline_aug.pt`
  - `efficientnet_b0_field_style_aug.pt`

## Notes

- Training both models will take significant time (several hours each)
- Ensure sufficient GPU memory (RTX 3070 Super or equivalent recommended)
- Mixed precision training (FP16) is enabled if available
- The experiment uses only PlantVillage data for training - no field data is used during training

