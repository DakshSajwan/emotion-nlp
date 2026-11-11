# Live Demo Script, Emergency Fallback Plan & Presentation Outline

This document provides a complete guide for presenting the Emotion Detection project during an oral defense, live committee demonstration, or seminar. It is divided into three comprehensive sections:
1. **Timed 5-Minute Live Demonstration Script** (with exact clicks, inputs, spoken lines, and failure case discussion)
2. **Emergency Fallback & Troubleshooting Plan** (CLI commands, static assets, and offline contingency)
3. **10-Slide Presentation Outline** (bullet-point presentation script grounded exclusively in real numbers)

---

## Part 1: Timed 5-Minute Live Demonstration Script

**Pre-demo preparation**:
Launch the web application before the committee enters:
```bash
.venv/bin/python run_all.py app
```
Verify the browser opens to `http://localhost:8501`. Have the browser window maximized side-by-side with your terminal.

---

### [0:00 - 0:45] Introduction & Sidebar Overview

- **Action**: Direct the committee's attention to the Streamlit web application. Point to the **Sidebar**.
- **What to Click**: In the sidebar **Model Selector**, ensure **DistilBERT v2 (improved)** is selected (the promoted production model). Scroll down to show the sidebar metrics.
- **What to Say**:
  > *"Good morning, members of the committee. Today I am demonstrating our end-to-end emotion detection system. Our project benchmarks a classical linear pipeline—TF-IDF paired with Logistic Regression—against our baseline DistilBERT v1 and our calibrated, multi-source DistilBERT v2 transformer across six affective categories: sadness, joy, love, anger, fear, and surprise.*
  >
  > *In the sidebar, you can see our multi-model selector and metrics summary. DistilBERT v2 is our active production model, trained on multi-source GoEmotions data with label smoothing and calibrated via temperature scaling ($T = 0.6775$) and selective prediction ($0.90$ abstain threshold).*
  >
  > *Let's begin with Tab 1: Analyze, demonstrating both high-confidence prediction and our selective abstention feature."*

---

### [0:45 - 1:45] Tab 1: "Analyze" — Confident Detection & Selective Abstention ("Not Sure")

- **Action**: Click on **Tab 1: Analyze**.
- **Step 1: Confident Correct Example**:
  - **What to Click**: Click on the preset button labeled **"😊 Joy"** (or paste: `"I just found out that I got accepted into my dream university! I am absolutely thrilled and grateful!"`).
  - **What to Click**: Click **"⚡ Classify Emotion"**.
  - **What the Screen Shows**:
    - Emotion Badge: **😊 JOY**
    - **Confidence**: **99.02%** (`is_confident: True`).
    - Probability breakdown shows `joy` dominating at 0.9902 with negligible mass across other classes.
  - **What to Say**:
    > *"Here in Tab 1, we test single-text inference on DistilBERT v2. For this clearly joyous expression, the model outputs **Joy** with a calibrated confidence of **99.02%**. Because this exceeds our 0.90 threshold, `is_confident` is True and the system presents a primary confident classification.*
    >
    > *Now, let's test what happens when we feed an out-of-domain, unemotional neutral sentence."*

- **Step 2: Selective Abstention ("Not sure: low confidence") Example**:
  - **What to Type**: Enter an unemotional, logistical sentence into the text box:
    ```text
    The flight arrives at gate B12.
    ```
  - **What to Click**: Click **"⚡ Classify Emotion"**.
  - **What the Screen Shows**:
    - An Amber dashed alert card: **⚠️ Not sure: low confidence**.
    - Informational text: *"Prediction confidence (55.3%) is below the selective abstain threshold."*
    - Top 2 candidate emotions clearly listed:
      - 1. 😲 **Surprise**: 55.3%
      - 2. 😊 **Joy**: 27.3%
    - Rather than presenting a false single label, the interface explicitly abstains and flags uncertainty!
  - **What to Say**:
    > *"Notice this critical safety behavior! Unemotional text like flight arrivals has no true emotion. Standard uncalibrated neural networks blindly output a high-confidence guess. But DistilBERT v2, calibrated with temperature scaling and our 0.90 abstain threshold, recognizes that confidence (55.31%) is too low. Instead of presenting a misleading badge, the system abstains, displays 'Not sure: low confidence', and presents the top two candidate emotions for human review."*

---

### [1:45 - 2:45] Tab 3: "Compare Models" — Where DistilBERT Fixes a Baseline Mistake

- **Action**: Click on **Tab 3: Compare Models**.
- **What to Type**: In the text area, paste this real test set sentence from `reports/metrics/error_analysis.md` (Table 1.A, Sample #3):
  ```text
  i feel respected and secure where i can journey toward loving and be loved in return
  ```
- **What to Click**: Click **"Run Comparison"**.
- **What the Screen Shows**:
  - A yellow banner: **⚠️ Model Divergence Detected: Baseline predicts 'love', DistilBERT predicts 'joy'**.
  - **Left Column (Baseline TF-IDF + LR)**: Predicts **Love** with **97.8% confidence** (Wrong).
  - **Right Column (DistilBERT)**: Predicts **Joy** with **99.9% confidence** (Correct — ground truth is `joy`).
- **What to Say**:
  > *"This example perfectly demonstrates why deep contextual transformers are necessary. In this sentence from our test set, the ground-truth emotion is **Joy**—the speaker feels safe, respected, and happy.*
  >
  > *However, notice our classical baseline: it predicts **Love** with overwhelming **97.8% confidence**. Why? Because the sentence contains the words 'loving' and 'loved'. The bag-of-words model isolates those n-grams and blindly triggers on the 'love' class.*
  >
  > *DistilBERT, however, uses multi-head self-attention. It connects 'loving and be loved' back to the root subject 'i feel respected and secure', correctly classifying the holistic affective state as **Joy** with **99.9% confidence**.*
  >
  > *To prove that this isn't a black-box mystery, let's inspect the token attributions using LIME in Tab 2."*

---

### [2:45 - 3:45] Tab 2: "Explain" — LIME Explainability in Action

- **Action**: Click on **Tab 2: Explain**.
- **What to Type**: Paste a clean emotional test sentence:
  ```text
  im feeling rather rotten so im not very ambitious right now
  ```
- **What to Click**: Ensure **Perturbation Samples** is set to `500` (or `200` for rapid live rendering) and click **"Generate Explanation"**.
- **What the Screen Shows**:
  - Prediction: **Sadness** (99.9% confidence).
  - Embedded interactive HTML visualization showing green and red highlighted words.
  - Horizontal bar chart of feature weights.
- **What to Say**:
  > *"Here in Tab 2, we deploy Local Interpretable Model-agnostic Explanations (LIME). For this input, DistilBERT predicts **Sadness**.*
  >
  > *Under the hood, LIME perturbed this sentence 500 times by masking words and observing probability fluctuations. It then fitted an interpretable linear surrogate model.*
  >
  > *Looking at the horizontal bar chart, the word **'rotten'** contributed a massive **+0.676 weight** toward sadness, followed by **'feeling'** at **+0.550**. Conversely, words like 'ambitious' pushed negatively against sadness.*
  >
  > *This demonstrates model faithfulness: DistilBERT is making predictions based on genuine affective adjectives, rather than latching onto spurious tokens like 'now' or 'right'."*

---

### [3:45 - 4:30] Tab 3/Tab 1: Honest Failure Case & Error Discussion

- **Action**: Switch back to **Tab 1: Analyze** (or Tab 3). Ensure **DistilBERT v2 (improved)** is active.
- **What to Type**: Paste this real error instance from `reports/metrics/probe_distilbert_v2_test_errors.csv` (Row 2):
  ```text
  Visiting our favorite old café alone felt cold, hollow, and painfully heavy today.
  ```
- **What to Click**: Click **"⚡ Classify Emotion"**.
- **What the Screen Shows**:
  - **Ground Truth**: `sadness` (a solitary reflection of grief and heartache).
  - **DistilBERT v2 Prediction**: **😠 ANGER** with **97.82% confidence** (`is_confident: True`) [Failure].
  - **Probability Breakdown**: `anger: 0.9782`, `sadness: 0.0125`, `fear: 0.0031`.
- **What to Say (Honest Critical Discussion)**:
  > *"A rigorous engineering presentation must openly discuss failure cases. Here is an instructive failure mode for our improved model.*
  >
  > *In our probe test split, the ground-truth emotion is **Sadness**—a poignant reflection on solitude and memory. However, DistilBERT v2 misclassifies it as **Anger** with high confidence (**97.82%**).*
  >
  > *Why did this happen? The adjectives 'cold', 'hollow', and especially 'painfully heavy' carry physical somatic tension that the model frequently associates with resentment and bitterness rather than weeping sadness. Because the sentence does not use conventional weeping markers, the transformer conflated somatic pain with anger.*
  >
  > *This honest failure illustrates the fundamental limitation of projecting human emotional complexity onto six mutually exclusive discrete classes, and underscores why affective models should remain human-in-the-loop assistive tools rather than autonomous arbiters."*

---

### [4:30 - 5:00] Tab 5: "Results" & Conclusion

- **Action**: Click on **Tab 5: Results**.
- **What to Show**: Scroll through the live benchmark table, the row-normalized confusion matrices, and the per-class F1 grouped comparison bar chart.
- **What to Say**:
  > *"Finally, Tab 5 loads our finalized test metrics directly from `reports/`. On the full 2,000-sample test set:*
  > - *DistilBERT achieves **92.95% accuracy** and **88.32% macro-F1**, outperforming the classical baseline by **+4.9% macro-F1** with major gains on minority classes like Love (+6.49%) and Surprise (+5.65%).*
  > - *However, our classical baseline trained in **0.87 seconds**—530× faster than DistilBERT—and executes inference in **5.09 milliseconds per 1,000 samples**, which is **535× faster**.*
  >
  > *In Tab 4, we also provide a batch CSV processing interface capable of exporting thousands of predictions in seconds.*
  >
  > *Thank you. I am now open to your questions."*

---

## Part 2: Emergency Fallback & Troubleshooting Plan

If anything fails technically during the live demonstration, execute the following fallback procedures immediately without breaking presentation flow:

```
+----------------------------------------------------------------------------------------------------+
|                                    EMERGENCY DECISION TREE                                         |
+----------------------------------------------------------------------------------------------------+
                                      Is Streamlit responsive?
                                             /        \
                                          [YES]       [NO]
                                           /            \
                                    Run Live Demo    Fallback to CLI
                                     (Part 1)        (Scenario A)
                                                        │
                                            Did CLI return predictions?
                                                    /        \
                                                 [YES]       [NO]
                                                  /            \
                                          Show CLI Output   Walk through static
                                          (Scenario A)      reports/ & figures/
                                                            (Scenario B/C)
+----------------------------------------------------------------------------------------------------+
```

---

### Scenario A: Streamlit Crashes or Port 8501 is Busy
- **Symptoms**: Browser shows `Cannot connect to server` or `Address already in use: 8501`.
- **Immediate Recovery Command**:
  ```bash
  # Kill any orphaned Streamlit instances
  pkill -f streamlit
  # Launch cleanly on a fresh port
  .venv/bin/python -m streamlit run app.py --server.headless true --server.port 8502
  ```
  Navigate browser to `http://localhost:8502`.
- **Zero-Downtime CLI Fallback**:
  If the browser fails entirely, switch directly to your open terminal and run the official CLI inference tool:
  ```bash
  # 1. Test Baseline Model
  .venv/bin/python -m src.predict "I feel respected and secure where i can journey toward loving" --model baseline

  # 2. Test DistilBERT Model
  .venv/bin/python -m src.predict "I feel respected and secure where i can journey toward loving" --model distilbert
  ```
- **Spoken Pivot**:
  > *"While Streamlit reloads, we can execute predictions directly via our production CLI interface in `src/predict.py`. As you see in the terminal, the baseline predicts 'love' with 97.8% confidence, while DistilBERT correctly resolves the sentence to 'joy' with 99.9% confidence."*

---

### Scenario B: Model Files Missing or Fail to Load
- **Symptoms**: `FileNotFoundError: Baseline model not found at models/baseline.joblib`.
- **Immediate Recovery Command**:
  ```bash
  # Retrain baseline in under 1 second:
  .venv/bin/python run_all.py baseline
  ```
- **If DistilBERT weights are unavailable**:
  Point to the pre-computed predictions file in `reports/predictions/distilbert_test_preds.csv`.
- **Spoken Pivot**:
  > *"Our pipeline contract mandates saving all test predictions to standardized CSV files in `reports/predictions/`. We can inspect the exact pre-computed probability distribution for this sample directly from our benchmark logs."*

---

### Scenario C: Offline Defense (No Internet / Network Down)
- **Status**: The entire repository is **100% offline-ready**. All datasets, models, tokenizers, and dependencies are locally cached in `data/`, `models/`, and `.venv/`. No internet access is required.
- **Verification**:
  Run unit tests to prove system integrity offline:
  ```bash
  .venv/bin/python -m pytest -q
  # Output: 17 passed in ~7s
  ```

---

### Scenario D: Static Visual Fallback Reference Table
If screen-sharing or UI rendering is completely disabled, open and display the following static artifacts directly from VS Code / IDE:

| Demonstration Element | Primary Static Artifact | Fallback Data File |
| :--- | :--- | :--- |
| **Headline Benchmark** | [`reports/metrics/comparison_table.md`](file:///Users/daksh/NLP%20Project/reports/metrics/comparison_table.md) | [`reports/metrics/comparison_table.csv`](file:///Users/daksh/NLP%20Project/reports/metrics/comparison_table.csv) |
| **Per-Class F1 Comparison** | [`reports/figures/compare_per_class_f1.png`](file:///Users/daksh/NLP%20Project/reports/figures/compare_per_class_f1.png) | [`reports/metrics/distilbert_report.csv`](file:///Users/daksh/NLP%20Project/reports/metrics/distilbert_report.csv) |
| **Confusion Matrices** | [`reports/figures/baseline_confusion.png`](file:///Users/daksh/NLP%20Project/reports/figures/baseline_confusion.png)<br>[`reports/figures/distilbert_confusion.png`](file:///Users/daksh/NLP%20Project/reports/figures/distilbert_confusion.png) | Section 2 of [`reports/metrics/error_analysis.md`](file:///Users/daksh/NLP%20Project/reports/metrics/error_analysis.md) |
| **Training Curves** | [`reports/figures/distilbert_training_curves.png`](file:///Users/daksh/NLP%20Project/reports/figures/distilbert_training_curves.png) | [`reports/metrics/distilbert_train_log.json`](file:///Users/daksh/NLP%20Project/reports/metrics/distilbert_train_log.json) |
| **LIME Visualizations** | [`reports/figures/lime_example_1.html`](file:///Users/daksh/NLP%20Project/reports/figures/lime_example_1.html)<br>[`reports/figures/lime_example_1.png`](file:///Users/daksh/NLP%20Project/reports/figures/lime_example_1.png) | [`reports/figures/lime_example_2.png`](file:///Users/daksh/NLP%20Project/reports/figures/lime_example_2.png) |
| **Detailed Error Cases** | [`reports/metrics/error_analysis.md`](file:///Users/daksh/NLP%20Project/reports/metrics/error_analysis.md) | [`reports/metrics/distilbert_top_errors.csv`](file:///Users/daksh/NLP%20Project/reports/metrics/distilbert_top_errors.csv) |

---

## Part 3: 10-Slide Project Presentation Outline

Use these slides as your visual presentation deck. Every bullet point is grounded in actual numbers and code:

```
====================================================================================================
                                      10-SLIDE PRESENTATION DECK
====================================================================================================
```

### Slide 1: Title & Project Overview
- **Title**: Emotion Detection in Text: Classical NLP vs. Transformer Architecture with Explainable AI
- **Subtitle**: An Empirical Benchmark of TF-IDF + Logistic Regression vs. DistilBERT on CPU Architecture
- **Candidate Name**: NLP Project Research Team
- **Core Theme**: Evaluating the trade-off between statistical linear efficiency and deep contextual representation on imbalanced multi-class emotion classification.

---

### Slide 2: Problem Statement & Motivation
- **Objective**: Multi-class categorization of text across 6 basic affective states: *sadness*, *joy*, *love*, *anger*, *fear*, and *surprise*.
- **The Core Trade-Off**:
  - Deep Transformers achieve state-of-the-art accuracy, but require heavy computation, large memory footprints, and slow CPU latency.
  - Classical linear baselines run orders of magnitude faster, but struggle with syntactic negation, word-order inversion, and semantic overlap.
- **Key Research Question**: Where is deep contextual self-attention indispensable, and where do classical linear models remain practically viable?

---

### Slide 3: Dataset: `dair-ai/emotion` & Class Skew
- **Corpus**: 20,000 English Twitter messages curated via distant supervision hashtags.
- **Data Partitions**:
  - **Train**: 16,000 samples (80.0%)
  - **Validation**: 2,000 samples (10.0%)
  - **Test**: 2,000 samples (10.0%)
- **Severe Class Imbalance**:
  - Joy (33.80%, 6,761 samples) and Sadness (28.98%, 5,797 samples) comprise **62.8%** of the dataset.
  - Anger: 13.55% (2,709 samples), Fear: 11.87% (2,373 samples).
  - Minority classes: Love: 8.21% (1,641 samples) and Surprise: **3.59%** (719 samples).
- **Sequence Length Profile**:
  - Mean word count: **19.14 words**; Median: **17.0 words**; 95th percentile: **41.0 words**; Max: **66 words**.
  - Justifies fixing `MAX_LENGTH = 64` tokens (captures 99.9% of text without quadratic padding overhead).

---

### Slide 4: Methodology: Dual Preprocessing & Negation Rule
- **Dual Cleaning Strategy**:
  - **Classical (`heavy_clean`)**: Contraction expansion (`don't` $\rightarrow$ `do not`), lowercasing, punctuation stripping, and WordNet lemmatization.
  - **Neural (`minimal_clean`)**: Whitespace normalization and URL/mention stripping, strictly preserving natural capitalization and punctuation for transformer attention heads.
- **The Critical Negation Preservation Rule**:
  - Standard NLTK stopword removal deletes negation words (`not`, `no`, `never`, `nor`), causing catastrophic sentiment inversion (*"not happy"* $\rightarrow$ *"happy"*).
  - Our pipeline explicitly preserves all negation tokens in `src/preprocess.py`.

---

### Slide 5: Classical Baseline Architecture & Tuning
- **Feature Extraction**: `TfidfVectorizer(max_features=20000, ngram_range=(1, 1))`.
- **Classification**: Multinomial Logistic Regression solved via L-BFGS with `class_weight='balanced'`.
- **Hyperparameter Optimization**:
  - Grid search over $C \in \{0.5, 1.0, 5.0, 10.0\}$ and unigram vs. bigram features.
  - Evaluated on fixed validation split using `PredefinedSplit` to prevent data leakage.
  - **Optimal Hyperparameters**: $C = 5.0$, `ngram_range = (1, 1)` achieving validation macro-F1 of **0.8723**.
- **Model Size & Training Speed**: Trains in **0.868 seconds**; serialized footprint is **1.10 MB** (1.16 MB on disk).

---

### Slide 6: Transformer Model: DistilBERT Fine-Tuning
- **Base Architecture**: `distilbert-base-uncased` (6 transformer layers, 12 attention heads, hidden dimension 768, 66M parameters).
- **Fine-Tuning Regime**:
  - Optimizer: AdamW ($\text{lr} = 5\times 10^{-5}$, weight decay $0.01$).
  - Batch size: 32 on CPU across 3 epochs (1,500 total optimization steps).
  - Dynamic padding via `DataCollatorWithPadding` to minimize padding compute.
  - Early stopping checkpointing: `load_best_model_at_end=True` monitored on validation `macro_f1`.
- **Validation Progression**: Macro-F1 rose smoothly: Epoch 1 (0.9088) $\rightarrow$ Epoch 2 (0.9120) $\rightarrow$ Epoch 3 (**0.9303**).

---

### Slide 7: Quantitative Benchmark Results
*Head-to-head evaluation on the uncorrupted 2,000-sample test split (`reports/metrics/comparison_table.csv`):*

| Metric | Baseline (TF-IDF + LR) | DistilBERT (Fine-Tuned) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **88.25%** | **92.95%** | **+4.70%** |
| **Macro-F1** | **83.42%** | **88.32%** | **+4.90%** |
| **Weighted-F1** | **88.48%** | **92.84%** | **+4.36%** |
| **Macro Precision** | 81.85% | **90.61%** | **+8.76%** |
| **Macro Recall** | 85.58% | **86.85%** | **+1.27%** |
| **Training Duration** | **0.868 seconds** | 460.589 seconds (7.68 min) | **530× compute** |
| **Inference Latency (per 1k)** | **5.093 ms** | 2,725.152 ms | **535× faster (Baseline)** |
| **Model Footprint** | **1.10 MB** (1.16 MB on disk) | 256.12 MB | **233× smaller (Baseline)** |

- **Minority Class Gains**: DistilBERT boosted F1 on *Love* (+6.49%) and *Surprise* (+5.65%).

---

### Slide 8: Explainable AI: LIME Token Attribution
- **Methodology**: Local Interpretable Model-agnostic Explanations (`LimeTextExplainer`, 500 perturbations, exponential cosine kernel).
- **Defensive Engineering**: Implemented whitespace fallback protection in `src/explain.py:predict_proba` to prevent tokenizer crashes when LIME generates empty perturbation strings.
- **Empirical Token Attribution Findings**:
  - In true emotional text: LIME verified strong affective token weights (*"furious"* +0.741 toward anger, *"rotten"* +0.676 toward sadness, *"terrified"* +0.619 toward fear).
  - In false positive text: Pinpointed that *"happy"* exerted +0.879 weight toward joy, exposing lexical capture.

---

### Slide 9: Error Analysis: Discrepancies & Real Failure Case
- **Cross-Model Error Segmentation (2,000 test samples)**:
  - Both correct: **1,709 samples (85.45%)**.
  - Baseline wrong, DistilBERT correct: **150 samples (7.50%)** $\rightarrow$ Contextual resolution.
  - DistilBERT wrong, Baseline correct: **56 samples (2.80%)** $\rightarrow$ Over-parameterization on keywords.
  - Both wrong: **85 samples (4.25%)** $\rightarrow$ Intrinsic ambiguity and label noise.
- **Dominant Confusion Pair**: `joy` $\leftrightarrow$ `love` (Baseline: 48 joy $\rightarrow$ love; DistilBERT: 31 love $\rightarrow$ joy).
- **Real High-Confidence Failure**: *"whenever i put myself in others shoes and try to make the person happy"* (True: anger, DistilBERT predicted joy with 0.9995 confidence).

---

### Slide 10: Conclusion & Future Work
- **Summary**:
  - DistilBERT is the clear winner for nuanced, high-stakes emotion detection (+4.9% Macro-F1, +8.76% Precision).
  - The Baseline is an extraordinary engineering baseline for high-throughput, resource-constrained streaming (535× faster, 233× smaller).
- **Future Enhancements**:
  1. **INT8 Quantization & ONNX Runtime**: Reduce transformer latency from 2,725ms to <15ms per 1,000 samples.
  2. **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**: Train rank $r=8$ adapters to slash training memory by 90%.
  3. **Focal Loss**: Dynamically down-weight easy joy/sadness samples to force optimization on rare surprise/love classes.
  4. **Multi-Modal Prosody**: Ingest acoustic pitch and inflection to resolve verbal irony and sarcasm.
