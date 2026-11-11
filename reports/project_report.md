# Comparative Evaluation of Classical and Transformer-Based Architectures for Multi-Class Emotion Detection with Local Interpretability

**Author:** NLP Project Research Team  
**Date:** November 2026  
**Repository:** `emotion-nlp`  
**Dataset:** HuggingFace `dair-ai/emotion`  

---

## Abstract

Emotion classification in natural language text is a central objective within affective computing, supporting applications in human-computer interaction, sentiment surveillance, and conversational systems. This study investigates the empirical trade-offs between a computationally lightweight classical machine learning pipeline—comprising Term Frequency-Inverse Document Frequency (TF-IDF) feature representations coupled with balanced Logistic Regression—and a deep contextual transformer architecture (`distilbert-base-uncased`) evaluated across six canonical emotion categories: *sadness*, *joy*, *love*, *anger*, *fear*, and *surprise*. Using the 20,000-sample `dair-ai/emotion` benchmark, we examine classification accuracy, macro-averaged F1-scores, training efficiency, and inference latency on CPU architectures. Furthermore, we deploy Local Interpretable Model-agnostic Explanations (LIME) to assess token-level feature attribution. The fine-tuned DistilBERT transformer achieves superior classification performance with an accuracy of 92.95% (+4.70 percentage points over baseline) and a macro-averaged F1-score of 88.32% (+4.90 percentage points), demonstrating substantial gains on underrepresented affective categories including *love* (+6.49% F1) and *surprise* (+5.65% F1). Conversely, the tuned classical baseline achieves a highly competitive 88.25% test accuracy and an 83.42% macro-F1, requiring only 0.868 seconds for end-to-end model training (530× faster than DistilBERT's 460.59s) and achieving sub-millisecond per-sample inference latency (5.093 ms per 1,000 samples, 535× faster than DistilBERT's 2,725.15 ms). Granular error analysis across 2,000 test samples highlights persistent challenges in disambiguating semantically adjacent class pairs (*love* vs. *joy* and *sadness* vs. *anger*) as well as pragmatic verbal irony. Finally, we provide an interactive Streamlit application enabling live inference, multi-model comparison, batch CSV processing, and in-situ LIME visualization.

---

## 1. Introduction

Affective state estimation from textual corpora requires distinguishing nuanced psychological states that often share surface lexicons. Traditional sentiment analysis categorizes text along a coarse polarity dimension (positive, negative, neutral). In contrast, fine-grained emotion classification aims to detect discrete affective categories, typically grounded in psychological emotion taxonomies such as Ekman's basic emotions or Plutchik's psycho-evolutionary wheel.

Recent advances in natural language processing (NLP) have been dominated by pre-trained self-attention transformers. While large language models and deep bidirectional encoders frequently establish benchmark state-of-the-art results, practical deployments in edge computing, high-throughput streaming environments, and resource-constrained environments necessitate careful evaluation of computational efficiency. This project systematically compares a tuned classical linear baseline with DistilBERT on standard CPU infrastructure, quantifying where deep contextual representations are indispensable and where classical bag-of-words pipelines remain practically viable.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Formulation
Given an English input text sequence $x = (w_1, w_2, \dots, w_T)$, the task is formulated as a single-label multi-class classification problem over a discrete emotion label space:
$$\mathcal{Y} = \{\text{sadness}, \text{joy}, \text{love}, \text{anger}, \text{fear}, \text{surprise}\}$$
The classifier maps sequence $x$ to a posterior probability distribution $P(y = c \mid x)$ for $c \in \mathcal{Y}$, assigning the predicted class $\hat{y} = \arg\max_{c \in \mathcal{Y}} P(y = c \mid x)$.

### 2.2 Core Project Objectives
1. **Pipeline Engineering**: Construct clean, reproducible, cross-platform training and evaluation modules without external cloud GPU dependencies.
2. **Empirical Benchmarking**: Compare classical linear classification (TF-IDF + Logistic Regression) with fine-tuned DistilBERT on identical test splits.
3. **Interpretability**: Integrate LIME to inspect token-level contributions and detect whether predictions align with human semantic intuition or spurious statistical artifacts.
4. **Interactive Demonstration**: Build a 5-tab Streamlit application featuring live inference, side-by-side comparison, batch CSV analysis, and reports inspection.

---

## 3. Related Work

Early textual affect detection relied on curated lexical resources such as WordNet-Affect (Strapparava & Valitutti, 2004) and NRC Emotion Lexicon (Mohammad & Turney, 2013), matching occurrences of emotive seed words. While transparent, rule-based keyword approaches suffer from poor recall and fail to handle contextual shifts, negations, or syntactic modification.

With statistical NLP, linear classifiers (Naive Bayes, Support Vector Machines, and Logistic Regression) trained on n-gram TF-IDF representations became standard baselines (Pang et al., 2002). Saravia et al. (2018) introduced the `dair-ai/emotion` benchmark, establishing that context-aware deep neural networks significantly outperform isolated n-gram models on social media text.

Bidirectional Encoder Representations from Transformers (BERT; Devlin et al., 2019) revolutionized text classification by pre-training multi-head self-attention mechanisms over masked language modeling objectives. DistilBERT (Sanh et al., 2019) applies knowledge distillation during pre-training to compress BERT-base by 40% in parameter count while retaining 97% of its language understanding capabilities and running 60% faster, making it an optimal candidate for CPU deployment.

To interpret complex non-linear models, Ribeiro et al. (2016) proposed Local Interpretable Model-agnostic Explanations (LIME), approximating black-box decision boundaries in the local neighborhood of an input instance through perturbed sampling and linear surrogate estimation.

---

## 4. Dataset and Exploratory Data Analysis (EDA)

### 4.1 Dataset Description
The system is evaluated on the standard `dair-ai/emotion` corpus, consisting of 20,000 distinct Twitter status updates labeled via distant supervision (emotive Twitter hashtags mapped to emotion clusters) and curated for consistency.

### 4.2 Split Partitions and Label Distribution
As verified in `reports/metrics/class_distribution.csv`, the dataset is partitioned into canonical splits:
- **Training Set**: 16,000 samples (80.0%)
- **Validation Set**: 2,000 samples (10.0%)
- **Test Set**: 2,000 samples (10.0%)

The empirical distribution of emotion categories across splits is detailed in Table 1:

| Emotion Class | Train Split | Validation Split | Test Split | Total Corpus | Proportion (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sadness** | 4,666 | 550 | 581 | 5,797 | 28.98% |
| **Joy** | 5,362 | 704 | 695 | 6,761 | 33.80% |
| **Love** | 1,304 | 178 | 159 | 1,641 | 8.21% |
| **Anger** | 2,159 | 275 | 275 | 2,709 | 13.55% |
| **Fear** | 1,937 | 212 | 224 | 2,373 | 11.87% |
| **Surprise** | 572 | 81 | 66 | 719 | 3.59% |
| **Total** | **16,000** | **2,000** | **2,000** | **20,000** | **100.00%** |

*Table 1: Class frequency counts across splits in the dair-ai/emotion dataset.*

### 4.3 Text Length Analysis
Length analysis from `reports/figures/text_length_distribution.png` and automated corpus profiling yields:
- **Mean Word Count**: 19.14 words per sample.
- **Median Word Count**: 17.0 words per sample.
- **95th Percentile Word Count**: 41.0 words per sample.
- **Maximum Word Count**: 66 words.
- **Minimum Word Count**: 2 words.

The compact sentence distribution confirms that sequence truncation at `MAX_LENGTH = 64` tokens preserves virtually 100% of substantive lexical content while minimizing computational overhead in transformer self-attention tensors.

---

## 5. Methodology

### 5.1 Dual Preprocessing Strategy
Different model paradigms require fundamentally distinct preprocessing workflows:

1. **Heavy Cleaning (`src/preprocess.py:heavy_clean`)**:
   Designed for the sparse TF-IDF feature space:
   - Contraction expansion (`don't` $\rightarrow$ `do not`, `i'm` $\rightarrow$ `i am`, `can't` $\rightarrow$ `cannot`).
   - Removal of non-alphabetic noise and superfluous punctuation.
   - WordNet lemmatization with protected part-of-speech stop-guards.
   - **Critical Negation Preservation Rule**: Standard stopword removal strips negation markers (`not`, `no`, `never`, `nor`, `n't`), inverting affective polarity (e.g., *"not happy"* collapsing to *"happy"*). Our preprocessing explicitly preserves all negation tokens.

2. **Minimal Cleaning (`src/preprocess.py:minimal_clean`)**:
   Designed for DistilBERT:
   - Strips URLs and user mentions.
   - Collapses irregular whitespace.
   - Retains natural grammar, capitalization, and punctuation cues essential for subword attention heads.

### 5.2 Baseline Model: TF-IDF + Logistic Regression
- **Vectorization**: `TfidfVectorizer` extracting word unigrams and bigrams up to a vocabulary ceiling of 20,000 features.
- **Classification**: Multinomial Logistic Regression optimized via L-BFGS with balanced class weighting:
  $$w_c = \frac{N}{K \cdot N_c}$$
  where $N$ is total training samples, $K=6$ is the number of classes, and $N_c$ is class support.
- **Hyperparameter Optimization**: Systematic grid search over $C \in \{0.5, 1.0, 5.0, 10.0\}$ and $\text{ngram\_range} \in \{(1, 1), (1, 2)\}$ utilizing fixed validation fold cross-validation (`PredefinedSplit`).
- **Optimal Hyperparameters**: Regularization parameter $C = 5.0$ and unigram range $(1, 1)$ achieved the highest validation macro-F1 of 0.8723.

### 5.3 Deep Transformer: DistilBERT Sequence Classifier
- **Base Architecture**: `distilbert-base-uncased` consisting of 6 transformer encoder blocks, 12 self-attention heads per layer, hidden dimensionality $d=768$, and 66 million parameters.
- **Classification Head**: Linear projection layer with dropout ($p=0.2$) mapping the pooled `[CLS]` token representation to 6 logits.
- **Fine-Tuning Regime**:
  - Optimizer: AdamW with weight decay $0.01$.
  - Learning Rate: $\eta = 5 \times 10^{-5}$ with linear warmup.
  - Loss Function: Cross-Entropy loss over canonical emotion classes.
  - Batch Size: 32 per device on CPU across 3 epochs.
  - Checkpoint Selection: `load_best_model_at_end=True` monitored against validation macro-F1.

### 5.4 Explainability Framework (LIME)
LIME models local decision boundaries around an instance $x$ by perturbing tokens and observing probability shifts:
$$\xi(x) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
where $f$ is the black-box classifier, $g$ is a simple linear explanation model, $\pi_x(z)$ is an exponential kernel distance measuring perturbation proximity, and $\Omega(g)$ penalizes surrogate complexity. We implement a batched probability inference engine resilient to empty perturbation strings generated during word-masking routines.

### 5.5 Evaluation Metrics & Justification for Macro-F1
Given the substantial class skew (Joy and Sadness account for 62.8% of samples, whereas Surprise represents only 3.6%), standard classification Accuracy is fundamentally misleading: a naive majority-class classifier would achieve over 33% accuracy while failing entirely on minority emotions.

Therefore, **Macro-Averaged F1-Score** is designated as the primary benchmark metric:
$$\text{Macro-F1} = \frac{1}{|\mathcal{Y}|} \sum_{c \in \mathcal{Y}} F1_c = \frac{1}{|\mathcal{Y}|} \sum_{c \in \mathcal{Y}} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
Macro-F1 assigns equal weight to each emotion category regardless of frequency, holding models rigorously accountable for minority classes (*surprise*, *love*).

---

## 6. Experiments and Results

### 6.1 Benchmark Comparison Table
Table 2 details the empirical results recorded on the official 2,000-sample test set directly from `reports/metrics/baseline.json`, `reports/metrics/distilbert.json`, and `reports/metrics/comparison_table.csv`:

| Performance Metric | Baseline (TF-IDF + LR) | DistilBERT Transformer | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **88.25%** | **92.95%** | **+4.70%** |
| **Macro-F1** | **83.42%** | **88.32%** | **+4.90%** |
| **Weighted-F1** | **88.48%** | **92.84%** | **+4.36%** |
| **Macro Precision** | **81.85%** | **90.61%** | **+8.76%** |
| **Macro Recall** | **85.58%** | **86.85%** | **+1.27%** |
| **Training Duration** | **0.868 seconds** | **460.589 seconds (7.68 min)** | 530× compute |
| **Inference Latency (per 1,000 samples)** | **5.093 ms** | **2,725.152 ms** | 535× faster (Baseline) |
| **Total Test Inference Time (2k samples)** | **0.0102 seconds** | **5.4503 seconds** | — |
| **Model Disk Footprint** | **1.10 MB** (1.16 MB on disk) | **256.12 MB** | 233× smaller (Baseline) |
| **Training Epochs** | 1 (GridSearchCV refit) | 3 epochs (batch size 32) | — |

*Table 2: Quantitative benchmark results on the 2,000-sample test split.*

### 6.2 Per-Class Classification Report Comparison
From `reports/metrics/baseline_report.csv` and `reports/metrics/distilbert_report.csv`, the breakdown of class-specific precision, recall, and F1 scores is presented in Table 3:

| Emotion Class | Baseline Prec | Baseline Rec | Baseline F1 | DistilBERT Prec | DistilBERT Rec | DistilBERT F1 | F1 Delta | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sadness** | 0.9423 | 0.9002 | 0.9208 | **0.9640** | **0.9673** | **0.9656** | **+0.0448** | 581 |
| **Joy** | 0.9369 | 0.8978 | 0.9170 | **0.9435** | **0.9612** | **0.9522** | **+0.0352** | 695 |
| **Love** | 0.6904 | **0.8553** | 0.7640 | **0.8690** | 0.7925 | **0.8289** | **+0.0649** | 159 |
| **Anger** | 0.8636 | **0.8982** | 0.8806 | **0.9392** | **0.8982** | **0.9182** | **+0.0376** | 275 |
| **Fear** | **0.8605** | 0.8259 | 0.8428 | 0.8458 | **0.9554** | **0.8973** | **+0.0545** | 224 |
| **Surprise** | 0.6173 | **0.7576** | 0.6803 | **0.8750** | 0.6364 | **0.7368** | **+0.0565** | 66 |
| **Macro Avg** | 0.8185 | 0.8558 | 0.8342 | **0.9061** | **0.8685** | **0.8832** | **+0.0490** | 2,000 |
| **Weighted Avg** | 0.8897 | 0.8825 | 0.8848 | **0.9297** | **0.9295** | **0.9284** | **+0.0436** | 2,000 |

*Table 3: Head-to-head per-class classification report across models.*

### 6.3 Key Analytical Findings
1. **Consistent Transformer Superiority**: DistilBERT improves the F1-score across all six classes without exception. The largest absolute gains occur on the difficult minority classes: *Love* (+6.49%), *Surprise* (+5.65%), and *Fear* (+5.45%).
2. **Precision vs. Recall Dynamics**: Baseline compensates for class sparsity via balanced class weighting, yielding higher recall on *Love* (0.8553 vs. 0.7925) and *Surprise* (0.7576 vs. 0.6364), but suffers from lower precision (0.6904 and 0.6173) due to false-positive over-triggering. DistilBERT provides much tighter precision (0.8690 on *Love*, 0.8750 on *Surprise*).
3. **Extreme Efficiency Asymmetry**: The classical model trains in less than one second (0.868s) and executes predictions at nearly 200,000 samples per second on CPU (5.093 ms / 1,000 samples). DistilBERT requires 460.59s (~7.68 minutes) for 3 epochs and 2,725.15 ms / 1,000 samples, making Baseline 535× faster and 233× smaller on disk (1.10 MB vs. 256.12 MB; 221× smaller relative to 1.16 MB on-disk file size).

---

## 7. Error Analysis and Interpretability

### 7.1 Cross-Model Discrepancy Breakdown
From `reports/metrics/error_analysis.md`, the 2,000-sample test set segments into:
- **Both Models Correct**: 1,709 samples (85.45%)
- **Baseline Wrong, DistilBERT Correct**: 150 samples (7.50%)
- **DistilBERT Wrong, Baseline Correct**: 56 samples (2.80%)
- **Both Models Wrong**: 85 samples (4.25%)

DistilBERT successfully corrects nearly 3× more errors (150 instances) than it introduces relative to the baseline (56 instances).

### 7.2 Most Frequent Confusion Pairs
As quantified from test predictions:

**Baseline Top 3 Confused Pairs (True $\rightarrow$ Predicted):**
1. `joy` $\rightarrow$ `love` (48 instances)
2. `sadness` $\rightarrow$ `anger` (22 instances)
3. `love` $\rightarrow$ `joy` (18 instances)

**DistilBERT Top 3 Confused Pairs (True $\rightarrow$ Predicted):**
1. `love` $\rightarrow$ `joy` (31 instances)
2. `joy` $\rightarrow$ `love` (19 instances)
3. `surprise` $\rightarrow$ `fear` (18 instances)

Notably, semantic boundary confusion between `joy` and `love` remains the dominant error across both architectures, reflecting distant-supervision labeling noise in emotive hashtag assignment.

### 7.3 High-Confidence Error Inspection
1. **Semantic Conflation (*Joy* vs. *Love*)**:
   - Baseline Sample: *"i feel that passionate about"* $\rightarrow$ True: `joy`, Baseline Pred: `love` (Conf: 0.98), DistilBERT Pred: `joy` (Conf: 0.56, Correct).
   - Baseline Sample: *"i feel respected and secure where i can journey toward loving and be loved in return"* $\rightarrow$ True: `joy`, Baseline Pred: `love` (Conf: 0.98), DistilBERT Pred: `joy` (Conf: 1.00, Correct).
2. **Competing Negative Affective Cues (*Sadness* vs. *Anger*)**:
   - Sample: *"i work well with almost every client ive ever been in contact with because i know what it means to feel depressed angry frustrated irritated hopeless and apathetic..."* $\rightarrow$ True: `sadness`, Baseline Pred: `anger` (Conf: 0.97), DistilBERT Pred: `sadness` (Conf: 1.00, Correct).
3. **Real DistilBERT High-Confidence Error**:
   - Sample from `reports/metrics/distilbert_top_errors.csv`: *"whenever i put myself in others shoes and try to make the person happy"* $\rightarrow$ True: `anger`, DistilBERT Pred: `joy` (Confidence: 0.9995).

### 7.4 LIME Interpretability Findings
LIME visualizations (regenerated from the trained DistilBERT model into `reports/figures/lime_example_1.html` through `lime_example_5.html`) confirm:
- For the top DistilBERT misclassification (*"whenever i put myself in others shoes and try to make the person happy"*), token attribution demonstrates that the word *"happy"* contributes +0.879 weight toward `joy`, completely blinding the model to the context of self-sacrifice and irritation.
- In clean emotional text, LIME highlights strong affective tokens: *"furious"* (+0.741 toward `anger`), *"rotten"* (+0.676 toward `sadness`), *"terrified"* (+0.619 toward `fear`), and *"content"* / *"happy"* toward `joy`.

---

## 8. Robustness, Multi-Source Data Augmentation, and Calibration (DistilBERT v2)

### 8.1 Why the Original Test Set Is Optimistic
Standard evaluation on the `dair-ai/emotion` test set yields an apparently robust accuracy of 92.95% and macro-F1 of 88.32%. However, this evaluation is fundamentally optimistic due to in-distribution evaluation artifacts:
1. **Source Uniformity**: Both training and test partitions were harvested from Twitter using identical distant-supervision hashtag queries.
2. **Template Redundancy**: Over 80% of samples in `dair-ai/emotion` contain explicit self-declarative prefixes (e.g., *"i feel"*, *"i am feeling"*, *"i feel like"*). Models can exploit these syntactical templates as high-confidence shortcuts rather than understanding latent semantic emotion.
3. **Absence of Neutral / OOD Text**: The corpus assumes an artificial closed-world prior where every input sentence strictly conveys one of the six basic emotions, failing to assess how classifiers behave when confronted with mundane, unemotional, or domain-shifted prose.

### 8.2 Construction of the Independent Faculty Probe Benchmark
To rigorously evaluate out-of-distribution robustness and simulate realistic evaluation, an independent 144-sentence probe dataset was authored ([`data/probe_set.csv`](file:///Users/daksh/NLP%20Project/data/probe_set.csv)):
- **Partitioning**: Split cleanly into 72 `dev` samples and 72 `test` samples.
- **Emotion Classes**: Exactly 20 samples per canonical emotion (10 dev / 10 test across sadness, joy, love, anger, fear, surprise).
- **Out-of-Domain Neutral Sentences**: Exactly 24 neutral, factual sentences (12 dev / 12 test)—including transit updates, office notices, and objective statements—designed to test false-positive confidence.
- **Syntactic Constraints**: All sentences are strictly between 5 and 25 words in length with verified **zero lexical overlap** against the training, validation, or original test sets.
- **Limitations of the Probe Set**: The probe set is deliberately small ($N = 144$ total), manually constructed by the research team, and affective ground-truth labels inherently retain a degree of human subjectivity.

### 8.3 Dataset V2 Engineering (`data/train_v2.csv`)
To overcome lexical overfitting and template brittleness, [`src/data_v2.py`](file:///Users/daksh/NLP%20Project/src/data_v2.py) synthesized an expanded multi-source corpus ([`data/train_v2.csv`](file:///Users/daksh/NLP%20Project/data/train_v2.csv)) totaling 17,992 unique samples:
1. **Original Ingestion**: All 16,000 samples from the original `dair-ai/emotion` training split.
2. **GoEmotions Integration**: Reddit comments from Google's `GoEmotions` dataset, filtering single-label instances and projecting 27 granular categories onto our 6 canonical classes using `GOEMOTIONS_MAP` (e.g., *admiration*, *caring* $\rightarrow$ `love`; *annoyance*, *disapproval* $\rightarrow$ `anger`; *nervousness* $\rightarrow$ `fear`; *amusement*, *pride* $\rightarrow$ `joy`).
3. **Prefix-Stripping and Rephrasing**: Applied stochastic prefix-stripping on repetitive `"i feel"` prefixes, replacing them with natural sentence beginnings (*"honestly,"*, *"truly,"*, or directly starting with the affective clause) to decouple prediction from n-gram triggers.
4. **Zero-Leakage Guarantee**: Enforced strict deduplication against `val.csv`, `test.csv`, and `probe_set.csv`.

### 8.4 Model v2 Training, Label Smoothing & Calibration
- **Label Smoothing ($\alpha = 0.1$)**: Fine-tuning DistilBERT v2 with label smoothing cross-entropy regularized logit over-confidence, encouraging softer predictive distributions over ambiguous multi-label boundaries.
- **Post-Hoc Temperature Scaling ($T$)**: Following Guo et al. (2017), scalar temperature $T$ was fitted on logits from the original validation split ([`data/val.csv`](file:///Users/daksh/NLP%20Project/data/val.csv)) by minimizing negative log-likelihood (NLL). Optimal temperature was determined as **$T = 0.6775$**, reducing validation NLL from 0.1860 to 0.1424.
- **Expected Calibration Error (ECE)**: Evaluated across 10 confidence bins on `data/test.csv`, ECE dropped dramatically from **0.0814** (before scaling) to **0.0256** (after scaling)—a **68.5% improvement** in predictive calibration ([`reports/figures/distilbert_v2_calibration.png`](file:///Users/daksh/NLP%20Project/reports/figures/distilbert_v2_calibration.png)).
- **Selective Abstention Threshold**: Tuned strictly on the probe `dev` partition (evaluating candidate thresholds from 0.40 to 0.90 in increments of 0.05). Threshold **$\tau = 0.90$** was selected, answering 75.0% of emotional instances with 88.89% selective accuracy and rejecting 83.33% of out-of-domain neutral queries.

### 8.5 Comprehensive Model Audit and Promotion Verdict
The final evaluation across the original test set and independent probe `test` split confirms:

| Model | Original Test Accuracy | Original Test Macro-F1 | Probe-Test Accuracy | Probe-Test Macro-F1 | Falsely Confident Neutral Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LR)** | 0.8825 | 0.8342 | 0.4167 | 0.4259 | **0.0000** (0/12) |
| **DistilBERT v1** | 0.9295 | 0.8832 | 0.6333 | 0.5958 | 0.3333 (4/12) |
| **DistilBERT v2 (improved)** | **0.9295** | **0.8929** | **0.8667** | **0.8676** | **0.2500** (3/12) |

- **Confident-Only Subset Performance (DistilBERT v2, threshold = 0.90)**:
  - **Selective Coverage (% Answered)**: **80.00%** (48 of 60 probe test instances answered)
  - **Accuracy on Answered**: **87.50%** (42 of 48 correct)
  - **Macro-F1 on Answered**: **87.88%**
- **Formal Verdict**: In accordance with the project promotion governance rule, **DistilBERT v2 was promoted to active status** (`ACTIVE_DISTILBERT = "distilbert_v2"` in [`src/config.py`](file:///Users/daksh/NLP%20Project/src/config.py)): its probe-test macro-F1 of **0.8676** exceeded v1 (0.5958) by **+27.18 percentage points** while preserving original test set accuracy at **0.9295**.

---

## 9. Limitations and Ethical Considerations

### 9.1 Technical and Methodological Limitations
1. **Sarcasm and Verbal Irony**: Sarcastic expressions (e.g., *"Oh wonderful, another flat tire in the freezing rain!"*) mislead classifiers due to prominent positive surface tokens (*"wonderful"*), resulting in incorrect `joy` classifications. Modeling pragmatic incongruity requires conversational discourse history and broader pragmatic context.
2. **Mixed and Blended Affective States**: Human psychological states often involve simultaneous, conflicting emotions (e.g., bittersweet graduation or simultaneous grief and love). The single-label classification formulation with categorical cross-entropy forces a mutually exclusive winner, obscuring subtle affective duality.
3. **Neutral and Out-of-Domain Text**: Unemotional factual text (e.g., logistics, weather reports, meeting announcements) has no valid emotion category. Standard uncalibrated neural models force probability mass into emotion classes with high confidence. While DistilBERT v2's temperature scaling and 0.90 abstain threshold reduce false confidence, neutral handling remains a fundamental challenge in closed-world emotion taxonomies.
4. **Very Short Inputs**: Brief 1- to 2-word inputs (e.g., *"fine"*, *"well"*, *"sure"*) lack necessary syntactic disambiguation, leaving inference susceptible to training frequency priors.

### 9.2 Ethical and Societal Considerations
1. **Assistive Decision Support**: Affective NLP classifiers should serve purely as assistive analytical or prioritization tools and must never be deployed as autonomous diagnostic arbiters in clinical psychiatric environments.
2. **Demographic and Sociolinguistic Bias**: Social media corpora inherit linguistic dialects, generational slang, and cultural idioms that may lead to disparate classification accuracy across demographic groups.

---

## 10. Conclusion and Future Work

This project delivered an end-to-end emotion detection system comparing classical machine learning and deep transformer paradigms. The classical TF-IDF + Logistic Regression baseline delivers an 88.25% test accuracy and an 83.42% macro-F1 score with sub-second training and microsecond latency. However, qualitative error analysis proves that deeper contextual models are essential to disambiguate semantically adjacent emotion categories (*joy*/*love*, *sadness*/*anger*) and resolve complex syntactic negations.

### Planned Enhancements:
- **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**: Introduce low-rank adaptation to enable rapid fine-tuning on consumer CPU hardware.
- **Model Quantization**: Deploy 8-bit integer quantization (INT8 via ONNX Runtime) to achieve transformer latencies below 10 milliseconds.
- **Multimodal Integration**: Expand the architecture to ingest acoustic prosody and facial action units for robust sarcasm resolution.

---

## References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171–4186.
2. Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word-Emotion Association Lexicon. *Computational Intelligence*, 29(3), 436–465.
3. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. *Proceedings of EMNLP*, 79–86.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of ACM SIGKDD*, 1135–1144.
5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS Workshop on Energy Efficient Machine Learning*.
6. Saravia, E., Liu, H. C. T., Huang, Y. H., Wu, J., & Chen, Y. S. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. *Proceedings of EMNLP*, 3687–3697.
7. Strapparava, C., & Valitutti, A. (2004). WordNet-Affect: An Affective Extension of WordNet. *Proceedings of LREC*, 1083–1086.
