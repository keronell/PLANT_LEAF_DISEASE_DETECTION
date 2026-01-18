# 45-Minute Research Presentation: Improving Generalization of Deep Learning Models
## Plant Leaf Disease Detection

**Authors:** Vlad Pavlyuk, Ronen Shershnev  
**Date:** 19/01/2026  
**Seminar:** Machine Learning  
**Supervisor:** Dr. Yehudit Aperstein

---

## **PRESENTATION TIMELINE (45 minutes)**

| Section | Time | Slides |
|---------|------|--------|
| 1. Title & Research Question | 2 min | 2 |
| 2. Background & Motivation | 5 min | 3 |
| 3. Problem Statement | 8 min | 4 |
| 4. Supervised vs Unsupervised Learning | 12 min | 6 |
| 5. Methodology & Framework | 10 min | 5 |
| 6. Experimental Results | 5 min | 3 |
| 7. Analysis & Discussion | 2 min | 2 |
| **Q&A** | **1 min** | - |

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

**Result:** Models trained on studio images **fail** on field images

**Talking Points:**
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
Improve field image classification performance from <10% to >60% F1 score through domain adaptation.

**Specific Objectives:**
1. **Identify Domain Shift:** Quantify performance gap between studio and field datasets
2. **Develop Adaptation Strategies:** Design augmentation and training techniques
3. **Evaluate Approaches:** Compare baseline vs. enhanced fine-tuning methods
4. **Achieve Balance:** Maintain >99% performance on studio while improving field performance

**Success Criteria:**
- Maintain ≥99% F1 on PlantVillage (controlled)
- Achieve ≥60% F1 on Plant_doc (field/document)
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

# **SECTION 4: SUPERVISED vs UNSUPERVISED LEARNING (12 minutes)**

## Slide 10: Learning Paradigms Overview

**Content:**

```
Machine Learning Approaches
│
├── Supervised Learning (with labels)
│   ├── Classification: Predict category
│   └── Regression: Predict value
│
├── Unsupervised Learning (without labels)
│   ├── Clustering: Find groups
│   └── Dimensionality Reduction: Simplify data
│
└── Reinforcement Learning (reward-based)
```

**Our Project:** Supervised Learning (Classification)

**Talking Points:**
- Introduce the two main paradigms
- Position our work in supervised learning
- Explain we'll compare why supervised was chosen

**Visual:** Tree diagram or Venn diagram

---

## Slide 11: Supervised Learning in Our Project

**Content:**

**Definition:**
Learning from labeled examples - each image has a known disease class (0-38)

**Our Implementation:**
```
Labeled Images (X, y) → EfficientNet-B0 → Fine-tuning → Disease Classifier
49K training images    ImageNet pre-trained    12-15 epochs    39 classes
```

**Key Characteristics:**
- ✅ **Clear Objective:** Predict specific disease class
- ✅ **Labeled Data:** 39 known disease categories with annotations
- ✅ **Transfer Learning:** Leverage ImageNet pre-trained weights
- ✅ **Measurable Success:** F1 score, accuracy metrics
- ✅ **Interpretable:** Confusion matrices show class-wise performance

**Talking Points:**
- Explain supervised learning simply
- Show how it applies to our problem
- Emphasize advantages for classification tasks

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

## Slide 13: Unsupervised Learning - Why Not?

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
- No need to work without labels

❌ **Evaluation Challenges:**
- How do you measure unsupervised performance?
- No ground truth for clusters

**Talking Points:**
- Acknowledge unsupervised capabilities
- Clearly explain why it doesn't fit our problem
- Show that supervised is the right choice

**Visual:** Comparison table or diagram

---

## Slide 14: Supervised vs Unsupervised Comparison

**Content:**

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|-------------------|---------------------|
| **Data** | Labeled (X, y) | Unlabeled (X only) |
| **Output** | Disease class (0-38) | Clusters/groups |
| **Evaluation** | F1, Accuracy | Silhouette, Inertia |
| **Use Case** | Classification | Exploration |
| **Our Project** | ✅ Primary approach | ❌ Not applicable |
| **Example** | "Tomato Early Blight" | "Cluster A" |

**When to Use Each:**

**Supervised:**
- Known categories
- Labeled data available
- Need specific predictions
- Can measure accuracy

**Unsupervised:**
- Unknown patterns
- No labels available
- Exploratory analysis
- Feature learning

**Talking Points:**
- Side-by-side comparison
- When each is appropriate
- Reinforce why supervised for our case

**Visual:** Comparison table (highlight supervised column)

---

## Slide 15: Could Unsupervised Help? (Future Work)

**Content:**

**Potential Hybrid Approaches:**

**1. Unsupervised Pre-training:**
- Use unlabeled field images for feature learning
- Then fine-tune with labeled data (semi-supervised)

**2. Anomaly Detection:**
- Identify unknown/new diseases not in 39 classes
- Alert when image doesn't match known patterns

**3. Data Exploration:**
- Cluster diseases before labeling
- Understand data structure
- Guide annotation strategy

**Why Not Now:**
- We have sufficient labeled data
- Focus is on domain adaptation (not feature learning)
- Classification task is well-defined

**Future Research:**
- Semi-supervised learning for expanding datasets
- Active learning for efficient annotation
- Unsupervised domain adaptation (no labels)

**Talking Points:**
- Show awareness of unsupervised potential
- Explain why not primary focus
- Suggest future directions
- Demonstrate deep understanding

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

**Experiment 1: Baseline Fine-Tuning**
- Combined datasets (PV + FieldPlant)
- Basic augmentation
- 5 epochs
- Standard learning rate

**Experiment 2: Quick Wins Strategy**
- Aggressive field augmentation
- 60/40 domain-balanced batches
- 12 epochs, layer-wise LR
- Weighted validation F1

**Experiment 3: Progressive Domain Adaptation**
- Progressive field ratio: 10% → 50%
- 5× field sample weight
- 15 epochs (early stopped at 12)
- Same aggressive augmentation

**Evaluation:**
- Per-dataset F1 scores
- Weighted F1 across domains
- Confusion matrices

**Talking Points:**
- Clear experimental progression
- Explain rationale for each step
- Show systematic approach

**Visual:** Experiment timeline or flowchart

---

## Slide 20: Datasets & Evaluation

**Content:**

**Datasets Used:**

| Dataset | Train | Test | Domain | Purpose |
|---------|-------|------|--------|---------|
| PlantVillage | 49,179 | 6,148 | Studio | Main training + evaluation |
| Plant_doc | - | 504 | Field/Document | External validation |
| FieldPlant | 4,640 | 928 | Real-world field | Training + evaluation |

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

**Performance Comparison:**

| Experiment | Main F1 | Plant_doc F1 | FieldPlant F1 |
|------------|---------|--------------|---------------|
| **Baseline** | 98.9% | 6.9% ❌ | 2.9% ❌ |
| **Quick Wins** | 99.7% | 59.7% ✅ | 86.0% ✅ |
| **Progressive** | 99.8% | 61.3% ✅ | 83.2% ✅ |

**Key Achievements:**
- ✅ Maintained 99.7%+ on controlled data (main)
- ✅ Improved field performance from <10% to 60%+ (Plant_doc)
- ✅ Achieved 83-86% on real-world field (FieldPlant)
- ✅ 790-2,830% improvement over baseline

**Talking Points:**
- Present results clearly
- Highlight improvements
- Emphasize successful domain adaptation

**Visual:** Results table with color coding (red/green)

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

## Slide 24: Why Supervised Learning Succeeded

**Content:**

**Success Factors:**

**1. Clear Problem Definition:**
- 39 known disease classes
- Classification task (not exploration)
- Supervised perfect fit

**2. Labeled Data Availability:**
- Large PlantVillage dataset (49K images)
- Enable transfer learning
- Measurable progress

**3. Domain Adaptation Works:**
- Fine-tuning bridges domain gap
- Augmentation + sampling strategies
- Multi-domain validation

**4. Transfer Learning Power:**
- ImageNet pre-training provides strong features
- Fine-tuning adapts to plant domain
- Efficient learning from limited field data

**Takeaway:**
Supervised learning + domain adaptation = successful generalization

**Talking Points:**
- Reflect on why supervised was right choice
- Connect to results
- General insights

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
- 99.7% F1 on studio images
- 60% F1 on field/document images
- 83-86% F1 on real-world field images

**Future Directions:**
- Semi-supervised learning (unlabeled field data)
- Active learning for annotation efficiency
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
- **15 min:** Finished problem statement, starting supervised vs unsupervised
- **27 min:** Completed supervised vs unsupervised, starting methodology
- **37 min:** Finished methodology, presenting results
- **42 min:** Completed results, starting conclusions
- **44 min:** Wrapping up, preparing for Q&A

## **Key Talking Tips:**

1. **Supervised vs Unsupervised Section:**
   - Take time to explain clearly (12 minutes)
   - Use examples: "Supervised gives you 'Tomato Early Blight', unsupervised gives you 'Cluster A'"
   - Connect to your project: "We need specific predictions, so supervised"

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
