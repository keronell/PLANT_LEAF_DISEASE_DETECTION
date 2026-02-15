# Plant Leaf Disease Detection: Domain Adaptation for Lab-to-Field Generalization

Code accompanying the article *"Improving generalization of deep learning models for plant leaf disease detection"* (Pavliyuk & Shershnev).

## Overview

This project reproduces experiments on domain shift between controlled laboratory imagery (PlantVillage) and real-world field imagery (Plant_doc, FieldPlant). It evaluates supervised domain adaptation (with labeled field data) and teacher–student pseudo-labeling (without labels).

## Requirements

- Python 3.8+
- PyTorch, torchvision, timm
- scikit-learn, PIL, pandas, matplotlib, seaborn

## Datasets

Place datasets under `data/`:

- **PlantVillage** (`Plant_leaf_diseases_dataset_with_augmentation/`): 61,486 images, 39 classes. [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease/data)
- **Plant_doc**: External field-like images, 27 classes. Download and structure as `data/Plant_doc/train/`, `data/Plant_doc/test/`.
- **FieldPlant** (`FieldPlant_reformatted/`): 5,170 field images. [Kaggle](https://www.kaggle.com/datasets/bloox2/fieldplant) — reformat to folder-per-class structure.

## Project Structure

```
├── data_labeling/          # Run first: builds metadata
│   └── 1_data_labeling.ipynb
├── experiment_1/           # Domain shift + naive fine-tuning (Article §4.1, §4.2.1)
│   ├── 4_test_common_classes.ipynb
│   └── 5_fine_tune_models.ipynb
├── experiment_2/           # Supervised adaptation: Quick Wins, Progressive (Article §4.2.2–4.2.3)
│   ├── 5.1_fine_tune_models_quick_wins.ipynb
│   └── 5.2_progressive_domain_adaptation.ipynb
├── experiment_3/           # Augmentation-only (Article §4.3, Table 5)
│   └── 1_train_augmentation_comparison.ipynb
├── experiment_4/           # Teacher–student pseudo-labeling (Article §4.4.1)
│   └── 1_teacher_student_pseudo_labeling.ipynb
├── experiment_5_threshold_ablation/  # Confidence threshold ablation (Article §4.4.2, Table 7)
│   └── 1_threshold_ablation.ipynb
├── data/
├── metadata/               # label_mapping.json, dataset_index.json (from data_labeling)
└── models/                 # Saved checkpoints
```

## Run Order

1. **data_labeling/1_data_labeling.ipynb** — Builds `metadata/label_mapping.json` and `metadata/dataset_index.json`. Run from project root.
2. **experiment_3** — Trains baseline and field-style models on PlantVillage; produces `efficientnet_b0_best.pt` and `efficientnet_b0_baseline_aug.pt` (required by later experiments).
3. **experiment_1** — Domain shift analysis and naive fine-tuning.
4. **experiment_2** — Quick Wins and Progressive domain adaptation (loads `efficientnet_b0_best.pt`).
5. **experiment_4** — Teacher–student pseudo-labeling (uses `efficientnet_b0_baseline_aug.pt` from experiment_3).
6. **experiment_5_threshold_ablation** — Pseudo-label threshold ablation (T=0.85, 0.90, 0.95).

## Article Experiments

| Section      | Experiment              | Notebook(s)                                               |
|-------------|--------------------------|-----------------------------------------------------------|
| 4.1         | Baseline + domain shift  | experiment_1, experiment_3                               |
| 4.2.1       | Naive fine-tuning        | experiment_1/5_fine_tune_models                           |
| 4.2.2       | Quick Wins               | experiment_2/5.1_fine_tune_models_quick_wins              |
| 4.2.3       | Progressive adaptation   | experiment_2/5.2_progressive_domain_adaptation            |
| 4.3         | Augmentation-only        | experiment_3/1_train_augmentation_comparison              |
| 4.4.1       | Teacher–student          | experiment_4/1_teacher_student_pseudo_labeling            |
| 4.4.2       | Threshold ablation       | experiment_5_threshold_ablation/1_threshold_ablation      |

