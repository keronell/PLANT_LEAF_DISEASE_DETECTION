# Supervised Learning Framework

## Project: Plant Leaf Disease Detection

**Problem:** Classify 39 plant diseases from images  
**Approach:** Transfer learning + domain adaptation  
**Model:** EfficientNet-B0 (ImageNet → fine-tuned)

---

## Datasets

| Dataset | Samples | Domain |
|---------|---------|--------|
| PlantVillage (main) | 49K train / 6K test | Studio images |
| Plant_doc | 504 test | Field/document |
| FieldPlant | 4.6K train / 928 test | Real-world field |

---

## Supervised Learning Pipeline

```
Labeled Images → Pre-trained Model → Fine-tuning → Disease Classifier
     (X, y)         (EfficientNet-B0)     (12-15 epochs)    (39 classes)
```

**Key Components:**
1. **Labeled Data:** Images with disease class annotations
2. **Transfer Learning:** ImageNet pre-trained weights
3. **Fine-tuning:** Adapt to plant disease domain
4. **Classification:** Predict disease class (0-38)

---

## Experiments & Results

| Experiment | Main F1 | Plant_doc F1 | FieldPlant F1 |
|------------|---------|--------------|---------------|
| **Baseline** | 98.9% | 6.9% ❌ | 2.9% ❌ |
| **Quick Wins** | 99.7% | 59.7% ✅ | 86.0% ✅ |
| **Progressive** | 99.8% | 61.3% ✅ | 83.2% ✅ |

**Key Improvement:** Domain adaptation solved shift from studio to field images

---

## Why Supervised Learning?

✅ **Clear goal:** Predict specific disease class  
✅ **Labeled data available:** 39 known disease categories  
✅ **Measurable:** Accuracy, F1 score metrics  
✅ **Transfer learning:** Leveraged ImageNet pre-training  

**Not Unsupervised Because:**
- Need specific class predictions (not just clusters)
- Disease categories are well-defined
- Requires interpretable outputs for diagnosis

---

## Framework Components

**1. Data Processing**
- Domain-aware augmentation (aggressive for field images)
- Weighted sampling (3-5× field data weight)
- 60/40 PV/Field batch balance

**2. Model Training**
- EfficientNet-B0 base architecture
- Layer-wise learning rates
- Weighted validation F1 (multi-domain)

**3. Evaluation**
- Per-dataset F1 scores
- Confusion matrices
- 99.7%+ on main, 60%+ on field datasets

---

## Key Results

✅ **99.7% F1** on controlled environment (main)  
✅ **59-61% F1** on field/document images (Plant_doc)  
✅ **83-86% F1** on real-world field (FieldPlant)  
✅ **790-2,830% improvement** over baseline on field data

**Achievement:** Successfully adapted from studio to real-world field conditions
