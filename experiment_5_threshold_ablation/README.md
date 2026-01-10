# Experiment 5: Pseudo-Label Threshold Ablation (Target Rare Classes)

## Objective

This experiment quantifies how pseudo-label quality vs quantity affects domain adaptation performance. Specifically, it tests whether relaxing the confidence threshold can improve rare / currently-zero F1 classes (especially tomato diseases) without harming PlantVillage performance.

## Key Research Questions

1. **Quality vs Quantity**: Does higher pseudo-label confidence (T=0.95) or higher coverage (T=0.85) lead to better domain adaptation?
2. **Rare Class Improvement**: Can a more relaxed threshold help tomato classes move from F1=0 → F1>0?
3. **Trade-off Analysis**: What is the optimal balance between pseudo-label quality and quantity?

## Experimental Design

### Fixed Rules (Same as Experiment 4)

- **Same splits**: PlantVillage train/val/test (all classes), Field Test set (intersection classes only), Field-Unlabeled pool
- **Same architecture**: EfficientNet-B0
- **Same training recipe**: Student initialized from teacher, 70% PV / 30% pseudo-field sampling, LR=1e-4, max epochs=15
- **Only variable that changes**: Confidence threshold T ∈ {0.95, 0.90, 0.85}

### Thresholds Tested

- **T=0.95**: High precision, low coverage (strictest)
- **T=0.90**: Reference point (from Experiment 4)
- **T=0.85**: Higher coverage, more noise (more relaxed)

## Experimental Protocol

### Step 1: Define Thresholds

Three thresholds are tested:
- T=0.95 (generated in this experiment)
- T=0.90 (loaded from Experiment 4)
- T=0.85 (generated in this experiment)

### Step 2: Generate Pseudo-Labels for Each Threshold

For each threshold T ∈ {0.95, 0.85}:
- Run teacher inference on Field-Unlabeled pool
- Filter samples where max softmax probability ≥ T
- Save CSV: `pseudo_labeled_field_T095.csv` and `pseudo_labeled_field_T085.csv`
- Report statistics: total kept, retention rate, per-class counts (especially tomato classes)

### Step 3: Train Student Models

For each threshold T:
- Construct combined training data: PV labeled train + pseudo-labeled field set
- Train student with identical settings (same epochs, LR, sampling ratio)
- Save best checkpoint by PV validation macro-F1
- Output checkpoints: `student_T095.pt`, `student_T085.pt` (T=0.90 uses model from Experiment 4)

### Step 4: Evaluate All Models

For each model (Teacher, Student_T095, Student_T090, Student_T085):

**A) PlantVillage Test (all classes)**:
- Accuracy
- Macro-F1

**B) Field Test (intersection classes only)**:
- Accuracy
- Macro-F1
- Per-class F1 (especially tomato classes)
- Confusion matrix

### Step 5: Analysis and Report

Generate comprehensive report with:

1. **Pseudo-label statistics table**:
   - Threshold, retained count, retention rate
   - Per-class pseudo-label counts (focus on tomato classes)

2. **Performance comparison**:
   - PV and Field metrics for all models
   - Per-class F1 for target rare classes (tomato_healthy, tomato_tomato_mosaic_virus, tomato_tomato_yellow_leaf_curl_virus)

3. **Success criteria evaluation**:
   - **Primary success**: Field Macro-F1 improves over Teacher AND over T=0.90
   - **Secondary success**: At least one tomato class moves from F1=0 → F1>0
   - **Guardrail**: PV Macro-F1 drop must be ≤ 0.01 absolute

4. **Interpretation**:
   - If T=0.95 best: pseudo-label noise is harmful; quality dominates
   - If T=0.85 best: coverage dominates; more field data helps even if noisier
   - If T=0.85 hurts: noise overwhelms; need better teacher or per-class thresholds

## Files Generated

### Output Files in `experiment_5_threshold_ablation/`

- `pseudo_labeled_field_T095.csv`: Pseudo-labels for T=0.95
- `pseudo_labeled_field_T085.csv`: Pseudo-labels for T=0.85
- `confusion_matrices_comparison.png`: Side-by-side comparison of all confusion matrices
- `confusion_T095.png`, `confusion_T085.png`: Individual confusion matrices for each threshold
- `report.txt`: Comprehensive analysis report

### Model Checkpoints in `models/`

- `efficientnet_b0_student_T095.pt`: Student model trained with T=0.95
- `efficientnet_b0_student_T085.pt`: Student model trained with T=0.85
- `efficientnet_b0_student.pt`: Student model for T=0.90 (from Experiment 4)

## Running the Experiment

1. **Prerequisites**: Ensure Experiment 4 has been run (for T=0.90 results and teacher model)

2. **Open the notebook**: `1_threshold_ablation.ipynb`

3. **Run all cells**: The notebook will:
   - Load teacher model and Experiment 4 results
   - Generate pseudo-labels for T=0.95 and T=0.85
   - Train student models for each threshold
   - Evaluate all models on PV and Field test sets
   - Generate visualizations and comprehensive report

4. **Expected runtime**: 
   - Pseudo-label generation: ~5-10 minutes
   - Student training (T=0.95): ~1-2 hours per threshold
   - Student training (T=0.85): ~1-2 hours per threshold
   - Evaluation: ~10-15 minutes

## Key Features

1. **Reproducible Splits**: Uses same random seed (42) as Experiment 4 for field data split
2. **Comprehensive Evaluation**: Per-class F1 scores, confusion matrices, and comparison tables
3. **Automatic Analysis**: Success criteria evaluation and interpretation in report
4. **Focus on Rare Classes**: Special attention to tomato classes that previously had F1=0

## Expected Outcomes

### Success Scenarios

1. **T=0.95 performs best**:
   - Interpretation: High-quality pseudo-labels are crucial; noise harms adaptation
   - Implication: Need better teacher model or stricter filtering

2. **T=0.85 performs best**:
   - Interpretation: Coverage matters more; additional field data helps despite noise
   - Implication: Current teacher is good enough; can relax threshold

3. **T=0.90 remains optimal**:
   - Interpretation: Balanced trade-off between quality and quantity
   - Implication: Experiment 4 threshold choice was correct

### Failure Scenarios

1. **No threshold meets success criteria**:
   - Field F1 doesn't improve over Teacher
   - PV F1 drops too much (>0.01)
   - Interpretation: Pseudo-label noise overwhelms benefits
   - Recommendation: Need better teacher model or per-class adaptive thresholds

2. **All thresholds improve Field but harm PV**:
   - Interpretation: Domain adaptation succeeds but at cost of in-domain performance
   - Recommendation: Consider different adaptation strategies or stronger PV regularization

## Dependencies

- Same dependencies as Experiment 4
- Requires Experiment 4 to be completed first (for teacher model and T=0.90 results)

## Notes

- The experiment automatically loads T=0.90 results from Experiment 4 if available
- If Experiment 4 results are not found, it will generate T=0.90 pseudo-labels and train a new student model
- All models use the same teacher initialization for fair comparison
- The balanced sampler maintains 70% PV / 30% pseudo-field ratio regardless of pseudo-label count

