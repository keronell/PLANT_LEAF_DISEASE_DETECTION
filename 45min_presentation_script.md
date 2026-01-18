# 45-Minute Research Presentation: Improving Generalization of Deep Learning Models

## Plant Leaf Disease Detection

**Authors:** Vlad Pavlyuk, Ronen Shershnev
**Date:** 19/01/2026
**Seminar:** Machine Learning
**Supervisor:** Dr. Yehudit Aperstein

---

## **PRESENTATION TIMELINE (45 minutes)**

| Section                                | Time            | Slides |
| -------------------------------------- | --------------- | ------ |
| 1. Title & Research Question           | 2 min           | 2      |
| 2. Background & Motivation             | 5 min           | 3      |
| 3. Problem Statement                   | 8 min           | 4      |
| 4. Supervised vs Semi-Supervised Learning | 12 min       | 6      |
| 5. Methodology & Framework             | 10 min          | 5      |
| 6. Experimental Results                | 5 min           | 3      |
| 7. Analysis & Discussion               | 2 min           | 2      |
| **Q&A**                          | **1 min** | -      |

---

# **SECTION 1: TITLE & RESEARCH QUESTION (2 minutes)**

## Slide 1: Title Slide

**Content:**

```
IMPROVING GENERALIZATION OF DEEP LEARNING MODELS 
FOR PLANT LEAF DISEASE DETECTION

Vlad Pavlyuk & Ronen Shershnev
Seminar in Machine Learning
Dr. Yehudit Aperstein
19 January 2026
```

**Talking Points:**

- Introduce yourself and co-author
- State the seminar context
- Brief: "We'll present our research on making plant disease detection models work in real-world conditions"

---

## Slide 2: Research Question

**Content:**

```
RESEARCH QUESTION

How can we achieve robust, high-accuracy leaf-disease 
classification on real-world ("in-the-wild") images 
with noise, variable lighting, and complex backgrounds?
```

**Talking Points:**

- State the core research question clearly
- Emphasize "real-world" vs. "controlled environment"
- Preview the challenge: generalization from studio to field images

---

# **SECTION 2: BACKGROUND & MOTIVATION (5 minutes)**

## Slide 3: Why Plant Disease Detection Matters

**Content:**

**Global Impact:**

- Plant diseases cause **10-40% yield losses** annually
- Direct threat to food security for growing population
- Economic losses in billions of dollars

**Current Challenges:**

- ❌ Manual inspection: Time-consuming, expertise-dependent, not scalable
- ❌ Late detection: Reduces treatment effectiveness
- ❌ Chemical overuse: Environmental and economic costs

**Solution:**

- ✅ Automated detection enables early intervention
- ✅ Scalable across large farms
- ✅ Reduces pesticide use through targeted treatment

**Talking Points:**

- Connect to food security and sustainability
- Highlight scalability advantage of automated systems
- Set up the motivation for deep learning approach

**Visual:** Statistics graph or infographic showing yield loss data

---

## Slide 4: Why Deep Learning?

**Content:**

**Advantages:**

- CNNs outperform classical computer vision methods
  - Mohanty et al. (2016): 99.35% accuracy on PlantVillage dataset
- Can learn hierarchical features automatically
- Generalize well on controlled datasets
- State-of-the-art performance on image classification

**However:**

- ⚠️ Generalization to real-world conditions remains challenging
- ⚠️ Models trained on clean, studio images fail on field images
- ⚠️ This is the problem we address

**Talking Points:**

- Reference Mohanty et al. study (cited in your references)
- Acknowledge success on controlled data
- Introduce the generalization challenge as the gap we address

**Visual:** Before/after comparison (studio vs. field images)

---

## Slide 5: The Domain Shift Problem

**Content:**

**Training Data (PlantVillage):**

- ✅ Controlled studio environment
- ✅ Consistent lighting
- ✅ Clean backgrounds
- ✅ High image quality
- ✅ Standardized setup

**Real-World Conditions:**

- ❌ Variable outdoor lighting
- ❌ Complex natural backgrounds
- ❌ Variable image quality
- ❌ Different camera angles
- ❌ Weather conditions (shadows, overcast)

**Talking Points:**

- **Result:** Models trained on studio images **fail** on field images
- Visual example showing the difference
- This is the "domain shift" problem
- Preview solution: domain adaptation

**Visual:** Side-by-side images showing studio vs. field conditions

---

# **SECTION 3: PROBLEM STATEMENT (8 minutes)**

## Slide 6: Problem Statement

**Content:**

**Core Problem:**
Deep learning models achieve excellent performance on controlled datasets but fail to generalize to real-world field conditions due to domain shift.

**Specific Challenges:**

1. **Distribution Mismatch:** Training (studio) ≠ Test (field) data distribution
2. **Domain Shift:** Different visual characteristics (lighting, backgrounds, quality)
3. **Limited Field Data:** Less labeled field data available for training
4. **Performance Gap:** 99% studio accuracy → <10% field accuracy

**Research Goal:**
Develop domain adaptation strategies to bridge studio-to-field gap while maintaining performance on controlled data.

**Talking Points:**

- Clearly define the problem
- Explain domain shift concept
- State what we aim to achieve
- Set expectations for results

---

## Slide 7: Scope of the Study

**Content:**

**What We Study:**

- ✅ Multi-class disease classification (39 disease classes)
- ✅ Domain adaptation from studio to field environments
- ✅ Fine-tuning strategies for improved generalization
- ✅ Data augmentation techniques for field images

**What We Don't Study:**

- ❌ Unsupervised anomaly detection (separate approach)
- ❌ Disease severity estimation (classification only)
- ❌ Real-time deployment optimization (focus on accuracy)
- ❌ Multi-plant type detection (single crop focus)

**Datasets:**

- PlantVillage (main, controlled): 49K training, 6K test
- Plant_doc (field/document): 504 test samples
- FieldPlant (real-world field): 4.6K training, 928 test

**Talking Points:**

- Clearly define boundaries
- Justify focus on supervised classification
- Introduce datasets briefly

**Visual:** Dataset statistics table or diagram

---

## Slide 8: Research Objectives

**Content:**

**Primary Objective:**
Improve field image classification performance from <10% to >80% F1 score through domain adaptation.

**Specific Objectives:**

1. **Identify Domain Shift:** Quantify performance gap between studio and field datasets
2. **Develop Adaptation Strategies:** Design augmentation and training techniques
3. **Evaluate Approaches:** Compare baseline vs. enhanced fine-tuning methods
4. **Achieve Balance:** Maintain >99% performance on studio while improving field performance

**Success Criteria:**

- Maintain ≥99% F1 on PlantVillage (controlled)
- Achieve ≥80% F1 on Plant_doc (field/document)
- Achieve ≥80% F1 on FieldPlant (real-world field)
- Demonstrate reproducible methodology

**Talking Points:**

- Clear, measurable objectives
- Success criteria are specific
- Preview evaluation metrics

---

## Slide 9: Relevance & Contribution

**Content:**

**Why This Matters:**

- **Practical Impact:** Enables deployment in real agricultural settings
- **Research Contribution:** Advances domain adaptation for plant disease detection
- **Methodological Value:** Provides reproducible framework for others
- **Sustainability:** Supports precision agriculture and reduced chemical use

**Novel Aspects:**

- Multi-domain evaluation (3 datasets)
- Domain-aware augmentation strategies
- Weighted multi-domain validation
- Progressive domain adaptation approach

**Talking Points:**

- Connect to broader agricultural AI field
- Highlight what's new in your approach
- Explain why the research matters

---

# **SECTION 4: SUPERVISED vs SEMI-SUPERVISED LEARNING (12 minutes)**

## Slide 10: Learning Paradigms Overview

**Content:**

```
Machine Learning Approaches
│
├── Supervised Learning (fully labeled)
│   ├── Classification: Predict category
│   └── All data has labels (X, y)
│
├── Semi-Supervised Learning (labeled + unlabeled)
│   ├── Uses labeled AND unlabeled data
│   └── Pseudo-labeling, self-training
│
├── Unsupervised Learning (no labels)
│   ├── Clustering: Find groups
│   └── Dimensionality Reduction: Simplify data
│
└── Reinforcement Learning (reward-based)
```

**Our Project:** 
- **Experiments 1-3:** Supervised Learning (Classification)
- **Experiments 4-5:** Semi-Supervised Learning (Pseudo-labeling)

**Talking Points:**

- Introduce the learning paradigms
- Show we used BOTH supervised and semi-supervised
- Explain why we needed to explore semi-supervised

**Visual:** Tree diagram showing supervised and semi-supervised branches

---

## Slide 11: Supervised Learning (Experiments 1-3)

**Content:**

**Definition:**
Learning from labeled examples - each image has a known disease class (0-38)

**Experiments 1-2: Domain Adaptation Fine-Tuning**

```
Labeled Images (X, y) → EfficientNet-B0 → Fine-tuning → Disease Classifier
49K PV + 4.6K Field     ImageNet pre-trained    12-15 epochs    39 classes
```

**Experiment 3: Augmentation Comparison**
- Baseline vs Field-Style augmentation
- Trained on PlantVillage only (no field data)
- Tested generalization to field images

**Key Characteristics:**

- ✅ **Clear Objective:** Predict specific disease class
- ✅ **Labeled Data:** 39 known disease categories with annotations
- ✅ **Transfer Learning:** Leverage ImageNet pre-trained weights
- ✅ **Measurable Success:** F1 score, accuracy metrics
- ✅ **Results:** 99.7% F1 on studio, 60-86% F1 on field

**Talking Points:**

- Explain supervised learning approach
- Show Experiments 1-2 achieved strong results
- Note Experiment 3 showed augmentation alone insufficient

**Visual:** Flowchart showing supervised learning pipeline

---

## Slide 12: Why We Chose Supervised Learning

**Content:**

**Problem Characteristics:**

- ✅ **Known Categories:** 39 well-defined disease classes
- ✅ **Labeled Data Available:** Large annotated dataset (PlantVillage)
- ✅ **Specific Task:** Classify disease (not explore patterns)
- ✅ **Need for Precision:** Medical/agricultural diagnosis requires accuracy

**Supervised Approach Benefits:**

- **Predictions:** Direct class output (e.g., "Tomato Early Blight")
- **Evaluation:** Objective metrics (F1 = 0.99 means 99% correct)
- **Transfer Learning:** Pre-trained models speed up training
- **Interpretability:** Can show "why" model predicted a class

**Real-World Requirements:**

- Farmers need specific disease names (not "group A")
- Treatment decisions depend on accurate classification
- Regulatory requirements may need explainable outputs

**Talking Points:**

- Connect problem needs to supervised approach
- Explain why classification is necessary
- Show alignment with real-world application needs

---

## Slide 13: Semi-Supervised Learning (Experiments 4-5)

**Content:**

**The Challenge:**
- Limited labeled field data (only 4.6K training samples)
- Abundant unlabeled field images available
- Need to leverage unlabeled data for domain adaptation

**Solution: Teacher-Student Pseudo-Labeling**

```
Step 1: Train Teacher Model
  PlantVillage (labeled) → Teacher Model (99% F1 on PV)

Step 2: Generate Pseudo-Labels
  Unlabeled Field Images → Teacher Predictions → High-confidence labels
  Threshold: 0.90-0.95 (keep only confident predictions)

Step 3: Train Student Model
  PlantVillage (labeled) + Pseudo-labeled Field → Student Model
  70% PV / 30% pseudo-field per batch
```

**Experiment 4: Teacher-Student Framework**
- Teacher: Baseline model on PV only
- Pseudo-labels: 1,819 high-confidence field images (49% retention)
- Student: Trained on combined dataset

**Experiment 5: Threshold Ablation**
- Tested thresholds: T=0.85, 0.90, 0.95
- Found T=0.95 optimal (quality > quantity)

**Talking Points:**

- Explain why semi-supervised was needed
- Show teacher-student framework
- Connect to limited labeled field data problem

**Visual:** Teacher-student pipeline diagram

---

## Slide 14: Supervised vs Semi-Supervised Comparison

**Content:**

| Aspect                | Supervised (Exp 1-3)        | Semi-Supervised (Exp 4-5)      |
| --------------------- | --------------------------- | ------------------------------ |
| **Data**        | Labeled (X, y) only        | Labeled + Unlabeled (X, y + X) |
| **Field Data**  | Requires labeled field data | Uses unlabeled field images    |
| **Approach**    | Fine-tuning with labels     | Pseudo-labeling                |
| **Results**      | 99.7% PV, 60-86% Field      | 98.9% PV, 3.5% Field (limited) |
| **Best Use**    | When labeled field data available | When unlabeled data abundant |
| **Our Project** | ✅ Experiments 1-3         | ✅ Experiments 4-5            |

**Why We Used Both:**

**Supervised (Experiments 1-2):**
- ✅ Labeled field data available (4.6K samples)
- ✅ Achieved excellent results (86% F1 on FieldPlant)
- ✅ Primary successful approach

**Semi-Supervised (Experiments 4-5):**
- ✅ Explored leveraging unlabeled field images
- ✅ Teacher-student framework tested
- ⚠️ Limited improvement (3.5% Field F1 vs 86% supervised)
- ✅ Demonstrated pseudo-labeling feasibility

**Talking Points:**

- Compare both approaches used
- Show supervised was more successful
- Explain why semi-supervised was still valuable to explore

**Visual:** Comparison table with results

---

## Slide 15: Why Not Pure Unsupervised? (Comparison)

**Content:**

**What Unsupervised Learning Could Do:**

- **Clustering:** Group similar disease images together
- **Anomaly Detection:** Identify "unusual" images (unknown diseases)
- **Feature Learning:** Extract patterns without labels
- **Dimensionality Reduction:** Visualize high-dimensional image data

**Why Not Primary Approach:**

❌ **No Specific Predictions:**
- Unsupervised gives "Group 1" not "Tomato Early Blight"
- Farmers need disease names, not cluster labels

❌ **Categories Are Known:**
- We have 39 defined disease classes
- No need to discover unknown patterns

❌ **Labeled Data Available:**
- PlantVillage has 49K labeled images
- Can use supervised/semi-supervised instead

❌ **Evaluation Challenges:**
- How do you measure unsupervised performance?
- No ground truth for clusters

**Our Approach:**
- ✅ **Supervised:** When labeled data available (Experiments 1-2)
- ✅ **Semi-Supervised:** When unlabeled data abundant (Experiments 4-5)
- ❌ **Unsupervised:** Not needed - we have labels and need specific predictions

**Future Research:**
- Unsupervised pre-training for feature learning
- Anomaly detection for unknown diseases
- Active learning for efficient annotation

**Talking Points:**

- Acknowledge unsupervised capabilities
- Explain why supervised/semi-supervised fit better
- Show comprehensive understanding of all paradigms

**Visual:** Comparison of all three approaches

---

# **SECTION 5: METHODOLOGY & FRAMEWORK (10 minutes)**

## Slide 16: Overall Framework

**Content:**

```
┌─────────────────────────────────────────────────────────┐
│ 1. PRE-TRAINED MODEL (ImageNet)                        │
│    EfficientNet-B0                                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. DATASET PREPARATION                                 │
│    • PlantVillage (studio) + FieldPlant (field)        │
│    • Domain-aware augmentation                         │
│    • Weighted sampling                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. FINE-TUNING                                         │
│    • Layer-wise learning rates                         │
│    • Domain-balanced batches (60/40)                   │
│    • Multi-domain validation                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. DOMAIN-ADAPTED MODEL                                │
│    • Works on studio AND field images                  │
└─────────────────────────────────────────────────────────┘
```

**Talking Points:**

- Visual overview of entire pipeline
- Show flow from pre-trained to adapted model
- Preview detailed components

**Visual:** Flowchart diagram

---

## Slide 17: Model Architecture

**Content:**

**Base Model: EfficientNet-B0**

**Why EfficientNet?**

- Optimal accuracy/efficiency trade-off
- State-of-the-art on ImageNet
- Smaller than ResNet/VGG but similar accuracy
- Mobile-friendly (can deploy on edge devices)

**Architecture:**

- **Input:** 224×224 RGB images
- **Pre-training:** ImageNet weights (1000 classes)
- **Adaptation:** Replace head → 39 disease classes
- **Parameters:** ~5.4M (lightweight)

**Transfer Learning Strategy:**

```
ImageNet Features → Plant Disease Features → Disease Classifier
(General)          (Domain-specific)        (Task-specific)
```

**Fine-tuning:**

- Freeze early layers (general features)
- Train later layers (disease-specific)
- Layer-wise learning rates

**Talking Points:**

- Explain EfficientNet choice
- Describe transfer learning
- Show adaptation process

**Visual:** Model architecture diagram

---

## Slide 18: Domain Adaptation Techniques

**Content:**

**Strategy 1: Aggressive Field Augmentation**

**Goal:** Simulate real-world conditions during training

**Techniques:**

- Rotation (30°), perspective transform
- Gaussian blur, sharpness adjustment
- Enhanced color jitter (brightness, contrast, saturation)
- Random erasing, affine transforms

**Rationale:** Field images are more variable than studio images

---

**Strategy 2: Domain-Balanced Sampling**

**Batch Composition:** 60% PV (studio) / 40% Field

**Weighted Sampling:** Field samples 3-5× more likely

**Effect:** Model sees field conditions frequently while maintaining studio performance

---

**Strategy 3: Multi-Domain Validation**

**Weighted F1 Metric:**

- 40% PlantVillage (main)
- 30% Plant_doc (field/document)
- 30% FieldPlant (real-world)

**Ensures:** Balanced performance across all domains

**Talking Points:**

- Explain each technique clearly
- Connect to domain shift problem
- Show how they work together

**Visual:** Before/after augmentation examples, sampling illustration

---

## Slide 19: Experimental Design

**Content:**

**SUPERVISED LEARNING EXPERIMENTS:**

**Experiment 1: Baseline Fine-Tuning**
- Combined datasets (PV + FieldPlant labeled)
- Basic augmentation
- 5 epochs
- Standard learning rate

**Experiment 2: Quick Wins Strategy**
- Aggressive field augmentation
- 60/40 domain-balanced batches
- 12 epochs, layer-wise LR
- Weighted validation F1
- **Result:** 99.7% PV, 59.7% Plant_doc, 86.0% FieldPlant

**Experiment 3: Augmentation Comparison**
- Baseline vs Field-Style augmentation
- Trained on PV only (no field data)
- Tested generalization
- **Finding:** Field-style augmentation didn't improve generalization

**SEMI-SUPERVISED LEARNING EXPERIMENTS:**

**Experiment 4: Teacher-Student Pseudo-Labeling**
- Teacher: Baseline model on PV only
- Pseudo-labels: High-confidence (≥0.90) on unlabeled field images
- Student: Trained on PV + pseudo-labeled field (70/30 ratio)
- **Result:** 98.9% PV, 3.4% Field F1 (limited improvement)

**Experiment 5: Threshold Ablation**
- Tested confidence thresholds: 0.85, 0.90, 0.95
- **Finding:** T=0.95 optimal (quality > quantity)
- **Result:** Best Field F1 = 3.5% (still limited)

**Talking Points:**

- Show both supervised and semi-supervised experiments
- Explain progression from supervised to semi-supervised
- Note supervised achieved much better results

**Visual:** Experiment timeline showing both approaches

---

## Slide 20: Datasets & Evaluation

**Content:**

**Datasets Used:**

| Dataset      | Train  | Test  | Domain           | Purpose                    |
| ------------ | ------ | ----- | ---------------- | -------------------------- |
| PlantVillage | 49,179 | 6,148 | Studio           | Main training + evaluation |
| Plant_doc    | -      | 504   | Field/Document   | External validation        |
| FieldPlant   | 4,640  | 928   | Real-world field | Training + evaluation      |

**Total:** 39 disease classes across all datasets

**Evaluation Metrics:**

- **Macro F1 Score:** Average F1 across all classes
- **Accuracy:** Overall classification accuracy
- **Weighted F1:** Domain-weighted average (multi-domain balance)

**Test Strategy:**

- No Test-Time Augmentation (TTA found to degrade performance)
- Single-pass inference
- Per-dataset evaluation

**Talking Points:**

- Introduce datasets clearly
- Explain evaluation metrics
- Justify test strategy

**Visual:** Dataset statistics table

---

# **SECTION 6: EXPERIMENTAL RESULTS (5 minutes)**

## Slide 21: Results Overview

**Content:**

**SUPERVISED LEARNING RESULTS:**

| Experiment            | Main F1 | Plant_doc F1 | FieldPlant F1 | Approach        |
| --------------------- | ------- | ------------ | ------------- | --------------- |
| **Baseline**    | 98.9%   | 6.9% ❌      | 2.9% ❌       | Basic fine-tune |
| **Quick Wins**  | 99.7%   | 59.7% ✅     | 86.0% ✅      | Domain adapt     |
| **Progressive** | 99.8%   | 61.3% ✅     | 83.2% ✅      | Progressive     |

**SEMI-SUPERVISED LEARNING RESULTS:**

| Experiment            | Main F1 | Field F1     | Approach              |
| --------------------- | ------- | ------------ | --------------------- |
| **Teacher**     | 98.9%   | 3.1% ❌      | PV only (baseline)   |
| **Student T=0.90** | 98.9%   | 3.4% ⚠️     | Pseudo-labeling      |
| **Student T=0.95** | 98.9%   | 3.5% ⚠️     | Best threshold       |

**Key Findings:**

- ✅ **Supervised:** Achieved 60-86% F1 on field datasets (major success)
- ⚠️ **Semi-Supervised:** Limited improvement (3.5% vs 86% supervised)
- ✅ **Conclusion:** Supervised domain adaptation most effective

**Talking Points:**

- Present both supervised and semi-supervised results
- Show supervised achieved much better performance
- Explain why supervised was the primary successful approach

**Visual:** Results table comparing both approaches

---

## Slide 22: Domain Shift Solved

**Content:**

**Before Domain Adaptation:**

```
Studio-trained Model
  ↓
Field Image Input
  ↓
Result: 6.9% F1 ❌ (Almost random guessing)
```

**After Domain Adaptation:**

```
Domain-Adapted Model
  ↓
Field Image Input
  ↓
Result: 59.7% F1 ✅ (Useful performance)
```

**Improvement Breakdown:**

- **Plant_doc:** 6.9% → 59.7% (**+767% improvement**)
- **FieldPlant:** 2.9% → 86.0% (**+2,826% improvement**)
- **Main:** 98.9% → 99.7% (maintained excellence)

**What This Means:**

- Model now works in real-world conditions
- Can be deployed for field use
- Practical utility achieved

**Talking Points:**

- Visual comparison of before/after
- Quantify the improvement
- Discuss practical implications

**Visual:** Before/after diagram, improvement percentages

---

## Slide 23: Key Findings

**Content:**

**1. Domain Shift Identified:**

- Baseline: 99% studio → 3-7% field = severe shift
- Confirmed need for domain adaptation

**2. Aggressive Augmentation Works:**

- Field augmentation critical for generalization
- Simulates real-world variability during training

**3. Sampling Strategy Matters:**

- Domain-balanced batches (60/40) essential
- Weighted sampling (3-5× field) improves learning

**4. Multi-Domain Validation:**

- Weighted F1 prevents overfitting to single domain
- Ensures balanced performance

**5. Both Approaches Successful:**

- Quick Wins: Best FieldPlant (86% F1)
- Progressive: Best Plant_doc (61% F1)
- Choose based on priority dataset

**Talking Points:**

- Summarize key learnings
- Connect findings to methodology
- Show systematic understanding

**Visual:** Bullet points with icons

---

# **SECTION 7: ANALYSIS & DISCUSSION (2 minutes)**

## Slide 24: Why Supervised Outperformed Semi-Supervised

**Content:**

**Supervised Learning Success (Experiments 1-2):**

**1. Labeled Field Data Available:**
- 4.6K labeled field training samples
- Direct supervision signal
- No label noise

**2. Effective Domain Adaptation:**
- Aggressive field augmentation
- Domain-balanced sampling (60/40)
- Multi-domain validation
- **Result:** 86% F1 on FieldPlant

**3. Transfer Learning Power:**
- ImageNet pre-training
- Fine-tuning adapts efficiently
- Strong feature learning

**Semi-Supervised Limitations (Experiments 4-5):**

**1. Pseudo-Label Quality:**
- Teacher model has low field accuracy (3.1%)
- Pseudo-labels contain noise
- Limited improvement (3.5% F1)

**2. Limited Unlabeled Data Benefit:**
- Only 1,819-2,046 pseudo-labels retained
- Quality threshold filtering reduces coverage
- Not enough to overcome domain shift

**3. Teacher Model Weakness:**
- Teacher trained on PV only
- Poor field generalization
- Pseudo-labels inherit this weakness

**Takeaway:**
- **Supervised:** Labeled field data + domain adaptation = 86% F1 ✅
- **Semi-Supervised:** Pseudo-labels from weak teacher = 3.5% F1 ⚠️
- **Conclusion:** When labeled field data available, supervised is superior

**Talking Points:**

- Compare why supervised succeeded
- Explain semi-supervised limitations
- Show comprehensive understanding of both approaches

**Visual:** Comparison diagram

---

## Slide 25: Conclusions & Contributions

**Content:**

**Research Contributions:**

✅ **Methodological:**

- Domain-aware augmentation pipeline
- Multi-domain validation framework
- Progressive adaptation strategy

✅ **Practical:**

- Achieved deployable performance on field images
- Maintained accuracy on controlled data
- Reproducible methodology for others

✅ **Performance:**

- 99.7% F1 on studio images (supervised)
- 60% F1 on field/document images (supervised)
- 83-86% F1 on real-world field images (supervised)
- Explored semi-supervised approach (3.5% F1, limited success)

**Future Directions:**

- Improve semi-supervised: Better teacher model or co-training
- Active learning for efficient annotation
- Unsupervised pre-training for feature learning
- Real-time deployment optimization

**Talking Points:**

- Summarize contributions
- Position work in field
- Suggest future work

---

# **FINAL SLIDES**

## Slide 26: References

**Content:**

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419.
- Ahmad, A., Saraswat, D., & El Gamal, A. (2023). A survey on using deep learning techniques for plant disease diagnosis...
- Hassan, S. M., et al. (2022). A survey on different plant disease detection using machine learning techniques...
- Bouguettaya, A., et al. (2023). A survey on deep learning-based identification of plant and crop diseases from UAV-based aerial images...

**Talking Points:**

- Acknowledge prior work
- Position your contribution

---

## Slide 27: Thank You & Questions

**Content:**

```
Thank You!

Questions?

Contact:
Vlad Pavlyuk | Ronen Shershnev
Seminar in Machine Learning
Dr. Yehudit Aperstein
```

**Talking Points:**

- Thank audience
- Invite questions
- Provide contact if needed

---

# **PRESENTATION DELIVERY GUIDE**

## **Timing Checkpoints:**

- **2 min:** Finished introduction, starting background
- **7 min:** Completed background, moving to problem statement
- **15 min:** Finished problem statement, starting supervised vs semi-supervised
- **27 min:** Completed supervised vs semi-supervised, starting methodology
- **37 min:** Finished methodology, presenting results
- **42 min:** Completed results, starting conclusions
- **44 min:** Wrapping up, preparing for Q&A

## **Key Talking Tips:**

1. **Supervised vs Semi-Supervised Section:**

   - Take time to explain clearly (12 minutes)
   - Show you used BOTH approaches (Experiments 1-3 supervised, 4-5 semi-supervised)
   - Explain why supervised achieved better results (86% vs 3.5% F1)
   - Demonstrate comprehensive understanding of learning paradigms
2. **Results:**

   - Highlight the dramatic improvements
   - Show before/after comparison
   - Emphasize practical impact
3. **Visuals:**

   - Use images showing studio vs. field conditions
   - Confusion matrices for detailed analysis
   - Progress charts showing improvement over experiments

## **Engagement Strategies:**

- **Ask Question:** "How many think models work in real-world conditions?"
- **Show Examples:** Visual comparison of studio vs. field images
- **Tell Story:** Walk through the experimental progression
- **Connect to Impact:** Emphasize agricultural importance

---

**END OF PRESENTATION SCRIPT**
