# Comprehensive Viva Questions & Oral Exam Defense Guide

This guide contains **35 rigorous oral examination questions and concise, grounded answers** for defending this Natural Language Processing project. All answers reflect the actual implementation in `src/`, `app.py`, and `tests/`, utilizing real metrics recorded in `reports/metrics/` and empirical findings from `reports/metrics/error_analysis.md`.

---

## 📑 Table of Contents
1. [NLP Theory & Fundamentals (Q1–Q6)](#1-nlp-theory--fundamentals)
2. [Project & Pipeline Design (Q7–Q12)](#2-project--pipeline-design)
3. [Empirical Results & Error Analysis (Q13–Q18)](#3-empirical-results--error-analysis)
4. [Explainability & Interpretability (Q19–Q23)](#4-explainability--interpretability)
5. [Limitations, Deployment & Ethics (Q24–Q27)](#5-limitations-deployment--ethics)
6. [Future Improvements & Scaling (Q28–Q30)](#6-future-improvements--scaling)
7. [Tricky Viva Questions & Honest Defenses (Q31–Q35)](#7-tricky-viva-questions--honest-defenses)

---

## 1. NLP Theory & Fundamentals

### Q1: How does subword tokenization (WordPiece in DistilBERT) differ from word-level tokenization in the classical baseline, and how does it handle Out-of-Vocabulary (OOV) tokens?
**Answer**:
- **Word-Level Tokenization (Baseline)**: Splits text on whitespace and punctuation into discrete whole-word tokens (e.g., `unhappiness` $\rightarrow$ `['unhappiness']`). If a word was not observed during training within the 20,000 feature limit, it is mapped to an out-of-vocabulary token or discarded entirely, losing all semantic information.
- **WordPiece Subword Tokenization (DistilBERT)**: Breaks words into frequent character n-grams and morphological subword units (e.g., `unhappiness` $\rightarrow$ `['un', '##happi', '##ness']`). Prefix `##` denotes continuation within a word.
- **OOV Handling**: Any rare, misspelled, or novel word is broken down into constituent known subwords (or individual characters in the worst case). The model never encounters an unseen OOV token, preserving morphological roots and affixes.

### Q2: What is the mathematical formulation of TF-IDF, and what linguistic information does it lose?
**Answer**:
- **Formulation**: For a term $t$ in document $d$ within a corpus $D$:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}, \quad \text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  where term frequency reflects local prominence and inverse document frequency dampens globally ubiquitous words. Vectors are $L_2$-normalized: $v = \frac{v}{\|v\|_2}$.
- **Lost Information**:
  1. **Word Order & Syntax**: Bag-of-words treats sentences as unordered multisets; *"not happy, very sad"* and *"not sad, very happy"* yield near-identical vector representations.
  2. **Semantic Similarity**: Synonyms (e.g., *"furious"* and *"enraged"*) are assigned orthogonal dimensions, sharing zero vector overlap.

### Q3: How do static word embeddings differ from contextual transformer representations for emotional words?
**Answer**:
- **Static Embeddings (Word2Vec, GloVe)**: Map each word string to a single fixed vector in $\mathbb{R}^d$ regardless of context. The polysemous word *"terrific"* (which can mean exceptionally good or historically terrifying) has one static representation.
- **Contextual Representations (DistilBERT)**: Generate dynamic vectors $h_i \in \mathbb{R}^{768}$ by passing the token through 6 bidirectional self-attention encoder layers. The embedding of *"terrific"* in *"a terrific achievement"* attends to *"achievement"* (producing positive joy activation), whereas in *"a terrific explosion"* it attends to danger cues (shifting toward fear/surprise).

### Q4: Explain the mathematical mechanism of Scaled Dot-Product Attention. Why is the scaling factor $\frac{1}{\sqrt{d_k}}$ essential?
**Answer**:
- **Equation**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  where queries $Q$, keys $K$, and values $V$ are linear projections of input embeddings with projection dimension $d_k = 64$ ($d_{model} = 768$ divided across 12 heads).
- **Purpose of Scaling ($\frac{1}{\sqrt{d_k}}$)**: For large projection dimensions $d_k$, the dot products $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ grow large in magnitude (variance scales with $d_k$). Large values push the softmax function into regions of extremely small gradients ($\approx 0$), causing the vanishing gradient problem during backpropagation. Dividing by $\sqrt{d_k} = \sqrt{64} = 8$ stabilizes variance to 1.0, preserving gradient flow.

### Q5: Describe the architectural topology of `distilbert-base-uncased` and the role of the `[CLS]` token.
**Answer**:
- **Architecture**: A 6-layer bidirectional transformer encoder with 12 self-attention heads per layer, hidden dimensionality $d=768$, intermediate feed-forward dimension $d_{ff}=3072$, GELU activations, and ~66 million parameters.
- **Role of `[CLS]` Token**: Prepended as the first token of every sequence (`[CLS] text [SEP]`). Through all 6 self-attention layers, the `[CLS]` token attends to every token in the sequence. Its final output vector $h_{\text{[CLS]}} \in \mathbb{R}^{768}$ serves as an aggregate semantic representation of the whole sentence and is fed to a linear classification head (`Linear(768, 6)`) with dropout ($p=0.2$) to produce class logits.

### Q6: What is knowledge distillation, and how was DistilBERT trained from BERT-base?
**Answer**:
- **Concept**: A compression technique where a compact "student" network is trained to replicate the behavior and representation space of a larger "teacher" model.
- **DistilBERT Setup**: Student has 6 layers (initialized by taking 1 out of every 2 layers from 12-layer BERT-base).
- **Tri-Fold Training Objective**:
  1. **Distillation Loss ($\mathcal{L}_{ce}$)**: Soft cross-entropy matching the student's softmax probability distribution with the teacher's softened probabilities at temperature $T$:
     $$\mathcal{L}_{ce} = -\sum p_i(T) \log q_i(T), \quad p_i(T) = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$$
  2. **Masked Language Modeling Loss ($\mathcal{L}_{mlm}$)**: Standard cross-entropy on masked tokens against ground-truth labels.
  3. **Cosine Distance Loss ($\mathcal{L}_{cos}$)**: Minimizes angle between student and teacher hidden state vectors.

---

## 2. Project & Pipeline Design

### Q7: Why benchmark TF-IDF + Logistic Regression against DistilBERT rather than an RNN/LSTM or a commercial LLM API?
**Answer**:
- **Architectural Spectrum**: TF-IDF + Logistic Regression represents the absolute peak of classical bag-of-words efficiency, while DistilBERT represents contextual deep learning. Comparing them directly quantifies the exact empirical return-on-investment of neural attention.
- **Why Not LSTMs**: Bi-LSTMs process tokens sequentially ($O(N)$ sequential operations), precluding parallelization on CPU. Prior literature (Saravia et al., 2018; Sanh et al., 2019) proves that transformers consistently outperform recurrent networks in accuracy while evaluating faster during inference.
- **Why Not LLM APIs (GPT-4/Claude)**: Commercial APIs introduce network latency (500–2,000 ms), recurring financial costs, privacy/GDPR concerns for sensitive text, and non-deterministic behavior. Our system runs 100% locally on free, open-source libraries on standard CPUs.

### Q8: Why is Macro-Averaged F1 the primary metric rather than Accuracy or Weighted-F1?
**Answer**:
- **Severe Class Imbalance**: In `dair-ai/emotion` (20,000 samples), Joy (33.80%) and Sadness (28.98%) represent ~63% of the dataset, while Surprise represents only 3.59% (719 total samples; 66 in test set).
- **Failure of Accuracy**: A naive baseline predicting only Joy and Sadness would achieve ~63% accuracy while failing 100% of the time on minority classes.
- **Failure of Weighted-F1**: Weighted-F1 weights each class F1 by its support:
  $$\text{Weighted-F1} = \sum_{c} \frac{N_c}{N} F1_c$$
  This allows high scores on majority classes to mask catastrophic failures on minority classes.
- **Macro-F1**: Assigns equal weight ($\frac{1}{6}$) to every emotion:
  $$\text{Macro-F1} = \frac{1}{6}\sum_{c=1}^6 F1_c$$
  Models are penalized heavily if they fail on *surprise* or *love*.

### Q9: How did you select `MAX_LENGTH = 64`, and what would happen if you used BERT's default of 512?
**Answer**:
- **Empirical Grounding from EDA**: Our automated corpus profiling in `src/eda.py` revealed:
  - Mean word count: 19.14 words.
  - Median word count: 17.0 words.
  - 95th percentile: 41.0 words.
  - Maximum word count: 66 words.
- **Selection**: Setting `MAX_LENGTH = 64` tokens accommodates virtually 100% of substantive content without truncating meaningful emotional clauses.
- **Consequence of 512**: Transformer self-attention complexity scales quadratically ($O(L^2)$) with sequence length $L$. Padding every sample to 512 tokens would perform $(512 / 64)^2 = 64\times$ more self-attention matrix multiplications per batch, wasting CPU RAM and increasing training time from 7.68 minutes to several hours without any accuracy gain.

### Q10: How does `class_weight='balanced'` work mathematically in Logistic Regression, and why was it necessary?
**Answer**:
- **Mathematical Formula**:
  $$w_c = \frac{N}{K \cdot N_c}$$
  where $N=16,000$ (total training samples), $K=6$ (number of classes), and $N_c$ is training support for class $c$.
- **Values in Our Training Split**:
  - Joy ($N_c = 5,362$): $w_{\text{joy}} = \frac{16000}{6 \times 5362} = 0.50$ (loss penalized less).
  - Surprise ($N_c = 572$): $w_{\text{surprise}} = \frac{16000}{6 \times 572} = 4.66$ (loss penalized $9.3\times$ more heavily).
- **Necessity**: Without inverse frequency weighting, unregularized gradient descent prioritizes majority classes to rapidly minimize global loss, resulting in low recall on *surprise* and *love*.

### Q11: Why did you implement two separate preprocessing pipelines (`heavy_clean` vs. `minimal_clean`)?
**Answer**:
- **For Baseline (`heavy_clean`)**: Linear models cannot learn morphological variations without explicit vocabulary consolidation. Lowercasing, expanding contractions (`don't` $\rightarrow$ `do not`), stripping punctuation, and WordNet lemmatizing (`apples` $\rightarrow$ `apple`) maps disparate surface forms into identical indices within the 20,000-dimensional TF-IDF space.
- **For DistilBERT (`minimal_clean`)**: Deep transformers utilize subword WordPiece tokenizers trained on natural text. Capitalization (e.g., *"HELP"* vs *"help"*), question marks (*"Really?"* vs *"Really."*), and exclamation points convey strong emotional valence. Stripping them degrades the contextual representations computed by attention heads.

### Q12: What is the Negation Preservation Rule in `src/preprocess.py`, and what happens if standard stopword removal is applied?
**Answer**:
- **The Rule**: In `src/preprocess.py`, our stopword retrieval function explicitly removes negation words from NLTK's English stopword set:
  ```python
  NEGATION_WORDS = {"not", "no", "never", "n't", "nor"}
  _STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
  ```
- **Consequence of Naive Removal**: Standard NLTK stopword lists include words like *"not"*, *"no"*, and *"nor"*. Filtering them naively transforms *"i am not happy"* into *"happy"*, directly inverting emotional polarity from sadness/anger to joy. Our pipeline explicitly preserves all negation tokens.

---

## 3. Empirical Results & Error Analysis

### Q13: State the headline test set results for both models. What is the efficiency trade-off?
**Answer**:
Evaluating on the 2,000-sample test set (`reports/metrics/comparison_table.csv`):
- **Accuracy**: DistilBERT achieved **92.95%** vs. Baseline's **88.25%** (+4.70 percentage points).
- **Macro-F1**: DistilBERT achieved **88.32%** vs. Baseline's **83.42%** (+4.90 percentage points).
- **Weighted-F1**: DistilBERT achieved **92.84%** vs. Baseline's **88.48%** (+4.36 percentage points).
- **Training Duration**: Baseline trained in **0.868 seconds**; DistilBERT required **460.59 seconds (7.68 minutes)** $\rightarrow$ **530× compute multiplier**.
- **Inference Latency (per 1,000 samples)**: Baseline took **5.093 ms**; DistilBERT took **2,725.152 ms** $\rightarrow$ **Baseline is 535× faster**.
- **Model Footprint**: Baseline is **1.10 MB** (1.16 MB on disk); DistilBERT is **256.12 MB** $\rightarrow$ **Baseline is 233× smaller**.

### Q14: On which specific emotion classes did DistilBERT show the largest performance gains, and why?
**Answer**:
Referring to `reports/metrics/baseline_report.csv` and `distilbert_report.csv`:
1. **Love**: F1 rose from **0.7640 $\rightarrow$ 0.8289** (+0.0649 / **+6.49%**).
2. **Surprise**: F1 rose from **0.6803 $\rightarrow$ 0.7368** (+0.0565 / **+5.65%**).
3. **Fear**: F1 rose from **0.8428 $\rightarrow$ 0.8973** (+0.0545 / **+5.45%**).
- **Reason**: *Love* and *surprise* are the two smallest minority classes (159 and 66 test samples). Their lexical keywords frequently overlap with *joy* (e.g., *"loved the gift"* vs. *"in love"*) and *fear* (e.g., *"shocked"* vs. *"scared"*). DistilBERT's contextual attention resolves whether a word like *"passionate"* reflects affectionate love or enthusiastic joy, which isolated TF-IDF n-grams cannot disambiguate.

### Q15: Why does the Baseline achieve higher recall on *Love* (0.855 vs. 0.792) and *Surprise* (0.758 vs. 0.636) than DistilBERT, despite having a lower F1-score?
**Answer**:
- **The Mechanism**: The Baseline compensated for minority class sparsity using aggressive balanced class weights ($w_{\text{surprise}} \approx 4.66$). This shifted its decision threshold lower, aggressively predicting *surprise* and *love* whenever weak lexical cues appeared.
- **The Consequence**: While this captured more true positives (high recall: 0.8553 on *love*, 0.7576 on *surprise*), it produced a flood of false positives, crippling precision (**0.6904** on *love*, **0.6173** on *surprise*).
- **DistilBERT's Balance**: DistilBERT learned sharper, context-aware decision boundaries, achieving vastly superior precision (**0.8690** on *love*, **0.8750** on *surprise*), which drove a much higher harmonic mean (F1).

### Q16: What are the most frequent confusion pairs for both models on the test set?
**Answer**:
From `reports/metrics/error_analysis.md`:
- **Baseline Top Confusion Pairs (True $\rightarrow$ Predicted)**:
  1. `joy` $\rightarrow$ `love`: **48 instances** (shared lexical tokens like *"passionate"*, *"beloved"*, *"blessed"*).
  2. `sadness` $\rightarrow$ `anger`: **22 instances** (co-occurring negative complaint tokens).
  3. `love` $\rightarrow$ `joy`: **18 instances** (affectionate statements mistaken for happiness).
- **DistilBERT Top Confusion Pairs (True $\rightarrow$ Predicted)**:
  1. `love` $\rightarrow$ `joy`: **31 instances** (expressions of romantic/familial warmth classified as general happiness).
  2. `joy` $\rightarrow$ `love`: **19 instances**.
  3. `surprise` $\rightarrow$ `fear`: **18 instances** (startling, shocking events sharing physiological shock vocabulary).

### Q17: What characterizes the 85 test samples where both models failed simultaneously?
**Answer**:
From Section 1.C of `reports/metrics/error_analysis.md`:
1. **Intrinsically Overlapping Emotions**: Sentences expressing multi-faceted affect (e.g., *"i didn't mean to get angry... i just hated myself why i am like this"*, true: `sadness`, both models predicted `anger`).
2. **Ambiguous Social Media Annotations**: Tweets like *"i could feel his breath on me and smell the sweet scent of him"* (true: `joy`, both models predicted `love` with >0.88 confidence).
3. **Subtle Irony / Understatement**: Statements like *"i don't feel particularly agitated"* (true: `fear`, both predicted `anger`).
These 85 errors represent the Bayes error rate of the dataset—instances where human annotators themselves disagree.

### Q18: Walk through the top high-confidence error made by DistilBERT. Why did the model fail?
**Answer**:
- **Sample Text**: *"whenever i put myself in others shoes and try to make the person happy"* (`reports/metrics/distilbert_top_errors.csv`, row 2).
- **Ground Truth**: `anger` (speaker is frustrated about unreciprocated self-sacrifice).
- **DistilBERT Prediction**: `joy` with **0.9995 confidence**.
- **Root Cause (Verified via LIME)**: In `reports/figures/lime_example_1.html`, token attribution reveals the single word *"happy"* exerted a massive **+0.879 feature weight** toward `joy`. DistilBERT succumbed to lexical capture: the overwhelming positive polarity of *"happy"* blinded the self-attention heads to the subtle grievance implied by the introductory clause.

---

## 4. Explainability & Interpretability

### Q19: Explain the mathematical objective and operational steps of LIME.
**Answer**:
- **Mathematical Formulation**:
  $$\xi(x) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
  where $f$ is the black-box classifier (DistilBERT), $g \in \mathcal{G}$ is an interpretable linear surrogate model, $\pi_x(z) = \exp(-D(x,z)^2 / \sigma^2)$ is an exponential proximity kernel, and $\Omega(g)$ penalizes model complexity.
- **Operational Steps**:
  1. Take original input text sequence $x$.
  2. Generate $N=500$ perturbations $z'$ by randomly turning off words (binary masking).
  3. Query DistilBERT with all perturbations to obtain class probabilities $f(z)$.
  4. Compute cosine distances between original text and perturbations to weight each instance via $\pi_x(z)$.
  5. Fit a weighted ridge regression model selecting the top $K=8$ features. The regression coefficients represent word importance weights.

### Q20: How did you ensure `predict_proba` in `src/explain.py` does not crash during LIME sampling?
**Answer**:
- **The Bug**: LIME generates perturbations by randomly zeroing out word masks. On short sentences, random sampling frequently masks out 100% of the words, passing empty strings (`""`) or pure whitespace (`"   "`) to the prediction callback.
- **Our Defense**: In `src/explain.py:predict_proba`:
  ```python
  safe_texts = []
  for t in texts:
      cleaned = minimal_clean(t) if (t and isinstance(t, str)) else ""
      safe_texts.append(cleaned if cleaned.strip() else " ")
  ```
  Empty strings are replaced with a single space token `" "`. The tokenizer processes this valid whitespace token cleanly, returning valid uniform probability distributions without throwing PyTorch tensor shape errors.

### Q21: What local surrogate model does LIME fit, and why is this explanation strictly "local"?
**Answer**:
- **Surrogate Model**: LIME fits a weighted linear ridge regression model:
  $$g(z') = w_0 + \sum_{i=1}^K w_i z'_i$$
  where $z'_i \in \{0, 1\}$ indicates the presence or absence of word $i$.
- **Why It Is "Local"**: Deep neural decision boundaries are globally non-linear, multi-dimensional, and impossible to approximate with a single linear equation. However, within a tiny $\epsilon$-neighborhood around a specific sentence $x$ (enforced by the proximity kernel $\pi_x$), the decision boundary is locally flat. The weights $w_i$ are valid only for that specific sentence and cannot be generalized across the entire corpus.

### Q22: What do your LIME visualizations reveal about how DistilBERT makes correct predictions?
**Answer**:
In `reports/figures/lime_example_2.html` through `lime_example_5.html`:
- For *sadness* (*"im feeling rather rotten so im not very ambitious right now"*): The token *"rotten"* contributes **+0.676** weight toward `sadness`, while *"feeling"* contributes +0.550.
- For *anger* (*"i am feeling furious that they cancelled the flight..."*): The token *"furious"* contributes **+0.741** weight toward `anger`.
- For *fear* (*"i feel terrified walking alone in this dark and eerie alley"*): The token *"terrified"* contributes **+0.619** weight toward `fear`.
This proves DistilBERT learns genuine affective associations rather than relying on spurious tokens like *"they"* or *"the"*.

### Q23: What are the primary technical limitations of LIME?
**Answer**:
1. **Sampling Instability (Non-Determinism)**: Because LIME uses Monte Carlo random masking, feature weights fluctuate slightly between runs unless random seeds are fixed.
2. **Loss of Syntax & Word Order**: Perturbations replace words with blanks, creating ungrammatical fragments (e.g., *"i [ ] [ ] not happy"*) that fall outside the natural manifold of pre-trained language models.
3. **High Latency**: Explaining a single tweet requires running 500 forward passes through DistilBERT, taking ~1.5 to 2.5 seconds on CPU.
4. **Surrogate Mismatch**: If the true decision boundary around $x$ has high curvature, a linear surrogate provides an oversimplified approximation.

---

## 5. Limitations, Deployment & Ethics

### Q24: What is distant supervision, and what systematic bias does it introduce into `dair-ai/emotion`?
**Answer**:
- **Distant Supervision**: Instead of trained psychologists manually annotating tweets, Saravia et al. (2018) scraped Twitter messages containing specific affective hashtags (e.g., `#anger`, `#joy`, `#love`) and mapped the hashtags to 6 emotion clusters, stripping the hashtag from the text.
- **Systematic Bias**:
  1. **Hyperbolic Tone**: Social media users use exaggerated hashtags (e.g., tweeting *"waiting in line forever #furious"* about a trivial wait).
  2. **Polysemous Hashtags**: A hashtag like `#love` is applied casually to pizza, sports teams, or romantic partners, causing persistent false confusion between *love* and *joy*.
  3. **Demographic Skew**: Twitter users over-represent younger, English-speaking demographics, failing to capture formal or cross-cultural expressions of emotion.

### Q25: Why do both classical and transformer models fail on verbal irony and sarcasm?
**Answer**:
- **Pragmatic Incongruity**: Sarcasm relies on an intentional opposition between literal surface meaning and intended emotional state (e.g., *"Oh brilliant, another puncture in the freezing rain!"*).
- **Model Vulnerability**:
  - The Baseline matches unigram tokens like *"brilliant"*, predicting `joy`.
  - DistilBERT's self-attention captures local syntax but lacks real-world commonsense knowledge—it does not inherently know that tire punctures and freezing rain are painful experiences unless explicitly trained on contrastive pragmatic pairs.

### Q26: In a production system, when would you deploy the Baseline over DistilBERT?
**Answer**:
Deploy the **Baseline** when:
1. **Ultra-Low Latency / High Throughput**: Processing streaming firehoses (e.g., 200,000 tweets/sec) on standard server hardware without GPU acceleration. Baseline executes in **5.09 ms per 1,000 samples** vs. DistilBERT's **2,725.15 ms**.
2. **Extreme Memory Constraints**: Deploying on embedded devices, edge microcontrollers, or serverless functions (AWS Lambda) where memory ceilings are strict. Baseline is **1.10 MB** vs. DistilBERT's **256.12 MB**.
3. **Auditability**: Regulated environments where global coefficients must be directly verifiable by legal teams.

Deploy **DistilBERT** when:
- Categorization accuracy on nuanced, complex, or minority emotions (*love*, *surprise*) is paramount and latency < 50ms per request is acceptable.

### Q27: What ethical risks arise from deploying emotion detection models?
**Answer**:
1. **Diagnostic Overreach in Mental Health**: Classifiers must never be used as autonomous diagnostic authorities. Misclassifying suicidal ideation or clinical depression as transient sadness could delay life-saving medical care.
2. **Punitive Workplace / Proctorship Surveillance**: Deploying emotion classifiers to track employee satisfaction or student engagement during exams assumes facial or textual expressions map linearly to productivity, punishing neurodiverse individuals.
3. **Feedback Loops in Customer Service**: Automated triage routing angry customers to retention queues can be gamed by users who intentionally use abusive language to skip wait times.

---

## 6. Future Improvements & Scaling

### Q28: How would you address minority class imbalance beyond balanced class weights?
**Answer**:
1. **Targeted Data Augmentation**:
   - **Back-Translation**: Translate minority samples (*surprise*, *love*) from English $\rightarrow$ German/French $\rightarrow$ English to synthesize lexically diverse paraphrases.
   - **Contextual Token Replacement**: Use a masked language model (`roberta-base`) to substitute non-emotional words with context-appropriate synonyms.
2. **Focal Loss**: Replace standard cross-entropy loss with Focal Loss:
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Setting $\gamma = 2.0$ dynamically down-weights easy, well-classified majority samples (joy/sadness) and focuses backpropagation on hard minority examples.

### Q29: How would you optimize DistilBERT for sub-10ms CPU inference?
**Answer**:
1. **INT8 Post-Training Quantization**: Quantize model weights and activation tensors from 32-bit floating point (FP32) to 8-bit integers (INT8) using PyTorch or ONNX Runtime. This reduces model size from 256 MB to ~65 MB and yields a $2.5\times$ to $4\times$ CPU speedup.
2. **ONNX Runtime Engine**: Export the PyTorch computation graph to ONNX and run inference with OpenVINO or Intel oneDNN execution providers, which fuse linear and layer-norm operations.
3. **Structured Pruning**: Prune redundant self-attention heads that exhibit low attention entropy across test samples.

### Q30: What is Parameter-Efficient Fine-Tuning (PEFT/LoRA), and how would it benefit this project?
**Answer**:
- **Mechanism**: Low-Rank Adaptation (LoRA) freezes the pre-trained transformer weight matrices $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices:
  $$W = W_0 + \Delta W = W_0 + B \cdot A$$
  where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and rank $r \ll \min(d, k)$ (e.g., $r=8$).
- **Benefits**:
  1. Only ~0.1% to 1% of parameters are updated during training (~300,000 parameters vs. 66 million), slashing GPU/CPU memory consumption during backpropagation.
  2. Adapter weights require only ~2 MB of storage per task, allowing a single base model to host multiple classification heads simultaneously.

---

## 7. Tricky Viva Questions & Honest Defenses

### Q31: *"DistilBERT improves accuracy by only 4.7 percentage points (88.25% $\rightarrow$ 92.95%) and macro-F1 by 4.9 percentage points over a simple TF-IDF + Logistic Regression baseline, but takes 530× longer to train and runs 535× slower at inference. Was fine-tuning a transformer truly worth the massive compute cost?"*
**Answer**:
*"In aggregate terms, 4.7 percentage points may appear modest, but aggregate metrics obscure where the gains occur. 
On the minority classes that matter most, DistilBERT delivered substantial improvements: **+6.49% F1 on Love** and **+5.65% F1 on Surprise**, while slashing false positives across the board (DistilBERT's macro precision is **90.61% vs. Baseline's 81.85%**, a +8.76 percentage point increase). 

Furthermore, error analysis shows DistilBERT corrected **150 errors made by the baseline (7.5% of the test set)**, particularly complex syntax and competing affective cues. 

Whether this justifies the compute cost depends strictly on deployment context: for an offline batch system or conversational mental-health assistant where misclassifying user distress has severe consequences, a 5% gain on minority emotions is well worth the extra latency. For high-volume streaming triage, the baseline is clearly the superior engineering choice."*

---

### Q32: *"In your error analysis, there are 56 instances where the classical baseline was correct while DistilBERT was wrong. How is it possible for a simple linear model to outperform a 66-million-parameter deep transformer on those samples?"*
**Answer**:
*"This occurs due to **lexical over-parameterization vs. direct keyword matching**. 

When a tweet contains an explicit, unambiguous emotional keyword—such as *"beloved friends"* or *"saddened"*—the classical model's TF-IDF feature extractor fires directly on that high-weight unigram, assigning the correct class with high fidelity. 

DistilBERT, by contrast, distributes attention across the entire sequence. In sentences with complex subordinate clauses, secondary conversational filler, or self-deprecating humor, DistilBERT's self-attention heads can attend to secondary clauses that dilute the primary keyword's signal. In our error analysis (e.g., Section 1.B of `error_analysis.md`), several of these 56 instances involved tweets where DistilBERT over-indexed on conversational context while the simple baseline succeeded by relying strictly on the dominant emotional keyword."*

---

### Q33: *"Is the `dair-ai/emotion` Twitter corpus a valid representation of real-world human emotion, or does it merely teach models to associate emotion words with emotion hashtags?"*
**Answer**:
*"To be completely honest, it is a proxy, not a ground truth of human psychological state. 

Because the labels were collected via distant supervision using Twitter hashtags, the dataset inherits specific artifacts:
1. It reflects what social media users *publicly broadcast* about their feelings, which is filtered through social desirability bias and online slang.
2. It suffers from label noise: people use `#love` to describe a television show or `#anger` flippantly.
3. It forces human emotion into six discrete, mutually exclusive buckets, whereas genuine human psychology experiences blended, continuous affect (Plutchik's wheel).

The dataset is an effective benchmark for comparing NLP architectures on informal social media text, but deploying this model in a clinical or psychiatric setting without retraining on clinically validated clinical interview transcripts would be scientifically invalid."*

---

### Q34: *"Does LIME explain how DistilBERT actually computed its internal attention representations, or does it merely provide a plausible post-hoc story that satisfies a human observer?"*
**Answer**:
*"It provides a post-hoc local surrogate, **not a mechanistic explanation of internal attention weights**. 

LIME treats DistilBERT as a complete black box: it does not inspect self-attention tensors, query-key dot products, or hidden state activations. Instead, it perturbs inputs, observes output probability changes, and fits a linear surrogate. This means LIME answers the question: *'Which words, if removed, cause the largest change in output probability?'* 

It does not answer: *'How did layer 4's attention heads combine these tokens?'* Research in explainability (e.g., Wiegreffe & Pinter, 2019; Rudin, 2019) highlights that post-hoc explanations can be plausible without being faithful to the network's internal reasoning. We use LIME as an empirical verification tool to detect obvious feature capture (like the word 'happy' dominating an angry sentence), but we do not claim it reveals the internal mechanistic reasoning of the transformer."*

---

### Q35: *"Your baseline used `PredefinedSplit` on the validation set for tuning $C$, and DistilBERT used the same validation set for checkpoint selection. Did you overfit your hyperparameters to that specific validation partition?"*
**Answer**:
*"We took explicit architectural precautions to mitigate validation overfitting:
1. **Grid Search Sparsity**: In `src/baseline.py`, our grid search evaluated only 8 discrete parameter pairs ($C \in \{0.5, 1.0, 5.0, 10.0\}$ and unigram vs. bigram), preventing fine-grained hyperparameter hacking.
2. **Early Stopping & Low Epoch Count**: DistilBERT was fine-tuned for only 3 epochs with a small learning rate ($5\times 10^{-5}$) and weight decay ($0.01$). Validation macro-F1 progressed smoothly across epochs (0.9088 $\rightarrow$ 0.9120 $\rightarrow$ 0.9303) without the divergence characteristic of overfitting.
3. **Strictly Untouched Test Set**: Neither model had access to the 2,000-sample test split during preprocessing, tuning, or checkpoint selection. 

The fact that DistilBERT achieved **92.95% accuracy on the test set**—closely matching its **93.7% validation accuracy**—empirically proves that the models generalized cleanly to unseen data rather than memorizing the validation fold."*

---

### Q36: *"What is distribution shift, and why was the 92.95% accuracy on the original `dair-ai/emotion` test set considered optimistic?"*
**Answer**:
*"Distribution shift occurs when the joint distribution of inputs and labels $P_{\text{test}}(X, Y)$ differs from the training distribution $P_{\text{train}}(X, Y)$. 

The original test set score (92.95% accuracy) was optimistic because both train and test splits came from the exact same Twitter stream collected with identical hashtag filters. Both splits shared distinct superficial artifacts: over 80% of samples began with the template *"i feel"* or *"i am feeling"*, and sentences relied on informal social media idioms. When tested on out-of-distribution text—such as our independent probe set—DistilBERT v1's macro-F1 dropped from **0.8832 down to 0.5958**, revealing that the model had learned dataset-specific shortcuts rather than generalized emotional understanding."*

---

### Q37: *"Why did you choose Google's GoEmotions dataset to build Dataset V2 rather than generating synthetic text with an LLM?"*
**Answer**:
*"We selected GoEmotions for three grounded methodological reasons:
1. **Diverse Natural Human Expression**: GoEmotions contains 58,000 real Reddit comments authored by diverse users across multiple subreddits, introducing conversational variety, slang, and syntax that break Twitter-specific hashtag biases.
2. **High-Quality Expert Annotation**: GoEmotions was annotated by multiple human raters with rigorous curation and inter-annotator agreement metrics, avoiding the hallucination, repetitive phrasing, and synthetic artifacts common in zero-shot LLM generations.
3. **Reproducibility & Open Science**: Using a peer-reviewed, open-source dataset (Demszky et al., ACL 2020) ensures our data pipeline is fully deterministic, auditable, and reproducible without API costs or proprietary model dependencies."*

---

### Q38: *"What specific problems arise when mapping a 27-class taxonomy like GoEmotions onto the canonical 6-class space?"*
**Answer**:
*"Mapping a fine-grained 27-class taxonomy onto 6 broad categories introduces two major challenges:
1. **Semantic Compression & Asymmetry**: Fine emotions like *admiration*, *caring*, and *desire* were mapped to `love`, while *annoyance* and *disapproval* were mapped to `anger`. This collapses subtle social distinctions: feeling mild annoyance at a late bus is linguistically different from visceral rage, yet both receive the `anger` label.
2. **Ambiguous Multi-Label Discards**: GoEmotions allows multi-label annotations (e.g. a comment labeled simultaneously as *optimism* and *nervousness*). Mapping these would introduce contradictory supervised targets (`joy` and `fear`). To prevent label noise, we strictly filtered for single-label instances where the mapped class was unambiguous, discarding overlapping multi-label instances."*

---

### Q39: *"Explain the mathematical mechanism of Temperature Scaling. Why does it improve calibration without changing classification accuracy?"*
**Answer**:
*"Temperature scaling modifies the logit vector $\mathbf{z}$ by dividing by a learned scalar parameter $T > 0$ before applying softmax:
$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$
Because dividing by a constant $T > 0$ is a **strictly monotonic transformation**, the relative ranking of logits is completely preserved:
$$\arg\max_i (z_i / T) = \arg\max_i (z_i)$$
Therefore, the predicted label $\hat{y}$ does not change for any sample, ensuring that classification accuracy remains identical. 

However, when $T > 1$, it softens overconfident probabilities; when $T < 1$ (as in our model where $T = 0.6775$), it sharpens underconfident probabilities. By fitting $T$ on validation set negative log-likelihood (NLL), the output probabilities become well-calibrated, dropping Expected Calibration Error (ECE) from **0.0814 to 0.0256**."*

---

### Q40: *"How does confidence-based selective prediction (abstention) function, and how did you select the abstain threshold?"*
**Answer**:
*"Selective prediction allows a classifier to output a prediction only when its maximum posterior probability $\max_c P(y = c \mid x) \ge \tau$, and otherwise abstain (return `'Not sure: low confidence'`). This enables a system to trade coverage for precision.

To select $\tau$, we conducted a grid search from 0.40 to 0.90 (step 0.05) strictly on the probe `dev` partition. We optimized the objective: maximize accuracy on answered instances, subject to answering at least 70% of 6-class emotional rows, while rejecting as many out-of-domain neutral rows as possible. Threshold $\tau = 0.90$ was selected: it answered 75.0% of emotional sentences with 88.89% selective accuracy and rejected 83.33% of neutral inputs."*

---

### Q41: *"Why was it critical to maintain an independent probe dataset with separate 'dev' and 'test' splits instead of evaluating on random cross-validation folds?"*
**Answer**:
*"Standard $K$-fold cross-validation on `dair-ai/emotion` only evaluates in-distribution interpolation. An independent probe dataset tests out-of-distribution extrapolation against human-authored sentences with completely different sentence structures and vocabulary.

Critically, partitioning the probe set into **`dev`** (72 samples) and **`test`** (72 samples) preserved scientific integrity:
- The **`dev` split** was used exclusively to observe trade-offs and select the 0.90 abstain threshold.
- The **`test` split** remained completely locked until Prompt 26, when it was used once for the promotion audit. 
If we had evaluated and tuned the threshold on the same probe data, we would have overfitted to our probe set, invalidating our claims of generalizability."*

---

### Q42: *"Why might DistilBERT v2 score slightly lower on certain in-distribution metrics while demonstrating superior real-world robustness?"*
**Answer**:
*"This is a classic manifestation of the **robustness-accuracy trade-off** in machine learning (Tsipras et al., 2018).

DistilBERT v1 achieved 92.95% on the original test set partly because it learned to exploit the repetitive `"i feel"` templates and specific lexical shortcuts present throughout the Twitter dataset. When we trained DistilBERT v2 with label smoothing ($\alpha = 0.1$), stripped template prefixes, and introduced diverse Reddit text, we deliberately regularized the model against memorizing Twitter-specific shortcuts. 

While DistilBERT v2 matched v1's overall test accuracy (92.95%) and achieved a higher macro-F1 (0.8929 vs. 0.8832), a regularized model might trade minor certainty on stereotyped phrases in exchange for massive gains on unseen out-of-distribution phrasing—as shown by v2's **+27.18 percentage point leap** in probe-test macro-F1."*

---

### Q43: *"What is the risk of 'overfitting to a probe set', and what experimental protocols did you employ to prevent it?"*
**Answer**:
*"Overfitting to an evaluation set occurs when researchers iteratively modify model architecture, training data, or hyperparameters to improve performance on a specific benchmark until the model merely memorizes the idiosyncrasies of that benchmark rather than learning generalized reasoning.

We prevented probe overfitting through three strict experimental guardrails:
1. **Zero-Leakage Hash Filtering**: Before training on `train_v2.csv`, all probe sentences were added to an exclusion set, guaranteeing that no probe sentence ever entered training.
2. **Strict Split Isolation**: Only the probe `dev` partition was used for threshold search; the probe `test` partition was never inspected or evaluated during model development.
3. **Fixed Model Parameters**: We did not iteratively alter DistilBERT v2's architecture or fine-tune hyperparameters to satisfy the probe set; the model was trained once, calibrated on the validation set, and evaluated on probe test in a single pass."*
