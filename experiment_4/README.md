# Experiment 4: Teacher-Student Pseudo-Labeling for Domain Adaptation

This experiment implements a teacher-student pseudo-labeling approach to improve generalization to field images using unlabeled field data.

## Overview

This experiment extends Experiment 3 by using a teacher-student framework with pseudo-labeling:

1. **Teacher Model**: Trained on PlantVillage only (using baseline augmentation from Experiment 3)
2. **Pseudo-Labeling**: Teacher generates high-confidence predictions on unlabeled field images
3. **Student Model**: Trained on combined PlantVillage + pseudo-labeled field data
4. **Evaluation**: Compare teacher vs student on both PV test and Field test sets

## Experimental Protocol

### Step 0: Freeze Evaluation Protocol
- Use the same Field Test Set from Experiment 3 (intersection classes only)
- Split remaining field images: 20% test, 80% unlabeled pool
- Field test images are never used for training or pseudo-label generation

### Step 1: Train Teacher Model
- Train baseline augmentation model on PlantVillage only
- Uses the same model from Experiment 3 if available
- Save best checkpoint by PV validation macro-F1
- Output: `teacher_model.pt`

### Step 2: Generate Pseudo-Labels
- Run inference using teacher on Field-Unlabeled Pool
- For each image, save predicted class and confidence score
- Keep only high-confidence predictions:
  - Start with threshold p ≥ 0.90
  - If too few samples, relax to 0.85, but not below 0.80
- Output: `pseudo_labeled_field.csv`

### Step 3: Build Combined Training Set
- Combine PlantVillage labeled training set (full PV)
- Add pseudo-labeled field set (filtered by confidence)
- Maintain 70% PV / 30% pseudo-field ratio per epoch
- Use lighter augmentation on pseudo-field data to avoid corrupting noisy labels

### Step 4: Train Student Model
- Initialize student from teacher weights
- Train on combined dataset with lower learning rate (1e-4 vs 3e-4)
- Early stopping using PV validation
- Output: `student_model.pt`

### Step 5: Evaluate and Report
- Evaluate Teacher vs Student on:
  - A) PV Test (all classes): Macro-F1, Accuracy
  - B) Field Test (intersection classes only): Macro-F1, Per-class F1, Confusion Matrix
- Success criterion: Student improves Field macro-F1 meaningfully without catastrophic PV collapse

## Files

- `1_teacher_student_pseudo_labeling.ipynb`: Main experiment notebook
- `report.txt`: Comprehensive results report (generated after running)
- `pseudo_labeled_field.csv`: Pseudo-labels with confidence scores
- `confusion_matrices_field_dataset.png`: Confusion matrices visualization
- `confidence_histogram.png`: Distribution of pseudo-label confidences

## Model Checkpoints

- `../models/efficientnet_b0_teacher.pt`: Teacher model (or uses baseline_aug from Experiment 3)
- `../models/efficientnet_b0_student.pt`: Student model trained on combined dataset

## Key Features

- **Confidence-based filtering**: Only high-confidence pseudo-labels are used (≥0.90, relaxable to 0.85/0.80)
- **Balanced sampling**: Maintains 70% PV / 30% pseudo-field ratio per epoch
- **Lighter augmentation on pseudo-labels**: Avoids corrupting potentially noisy labels
- **Lower learning rate for student**: 1e-4 vs teacher's 3e-4 to prevent overfitting to noisy labels
- **Comprehensive evaluation**: Full metrics on both PV and Field test sets

## Running the Experiment

1. Ensure Experiment 3 has been run (or teacher model exists)
2. Open `1_teacher_student_pseudo_labeling.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. The notebook will:
   - Load or train teacher model
   - Generate pseudo-labels on unlabeled field images
   - Train student model on combined dataset
   - Evaluate both models
   - Generate all reports and visualizations

## Expected Outcomes

### Success Criteria
- Student improves Field macro-F1 compared to teacher
- PV performance does not collapse (change < 0.05)
- Pseudo-labeling provides meaningful domain adaptation signal

### Interpretation
- **If successful**: Pseudo-labeling effectively leverages unlabeled field data for domain adaptation
- **If partial success**: Field improves but PV degrades - may need better confidence threshold or sampling strategy
- **If failure**: Pseudo-labels may be too noisy, or need different approach (e.g., co-training, consistency regularization)

## Notes

- The experiment automatically uses the teacher model from Experiment 3 if available
- Confidence threshold adapts if too few samples are retained
- Training the student model may take several hours
- All results are saved to `experiment_4/` directory

