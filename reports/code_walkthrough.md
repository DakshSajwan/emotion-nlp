# Project Code Walkthrough & Oral Exam Defense Guide

This guide is written for a student preparing to defend this Natural Language Processing project in an oral exam. It explains the entire architecture in plain, rigorous language, detailing how raw text flows into trained models, explainability engines, and interactive web applications, followed by a comprehensive, file-by-file breakdown of every module in `src/`, `app.py`, and `tests/`.

---

## 🗺️ System Architecture & Data Flow Diagram

The diagram below maps the complete lifecycle of data across the codebase, from raw corpus ingestion to live user inference and interactive queue sampling in the Streamlit web application:

```text
====================================================================================================
                                      RAW DATASET INGESTION
                                 HuggingFace "dair-ai/emotion"
                           (20,000 English tweets, 6 emotion labels)
                                        [src/data.py]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
              [Train: 16,000]                                 [Validation: 2,000]
              (80% of corpus)                                 (10% of corpus)
                      │                                               │
                      │                                               │
     ┌────────────────┴────────────────┐                              │
     │                                 │                              │
     ▼                                 ▼                              ▼
[CLASSICAL PIPELINE]          [NEURAL PIPELINE]             [VALIDATION FOLD]
 src/preprocess.py:            src/preprocess.py:            Used for baseline tuning
 heavy_clean()                 minimal_clean()               (GridSearchCV) & DistilBERT
 • Lowercase                   • Strip URLs & @mentions      early stopping checkpointing
 • Standardize apostrophes     • Collapse whitespace
 • Expand contractions         • Keep casing & syntax
 • Lemmatize (WordNet)
 • Preserve negations!
     │                                 │
     ▼                                 ▼
[TF-IDF VECTORIZATION]        [SUBWORD TOKENIZATION]
 TfidfVectorizer               AutoTokenizer
 (max_features=20,000,         ("distilbert-base-uncased",
  ngram_range=(1,1))            max_length=64, dynamic padding)
     │                                 │
     ▼                                 ▼
[LOGISTIC REGRESSION]         [DISTILBERT CLASSIFIER]
 C=5.0, class_weight='balanced' 6 transformer encoder layers,
 Solved via L-BFGS             66M parameters, AdamW (lr=5e-5)
 [src/baseline.py]             [src/transformer.py]
     │                                 │
     ▼                                 ▼
 models/baseline.joblib        models/distilbert/
 (1.10 MB, trains in 0.87s)    (256.12 MB, trains in 7.68m)
     │                                 │
     └────────────────┬────────────────┘
                      │
                      ▼
====================================================================================================
                                    UNIFIED TEST EVALUATION
                                  [Test Set: 2,000 samples]
                                      [src/evaluate.py]
                      │
                      ├─► reports/predictions/{baseline,distilbert}_test_preds.csv
                      ├─► reports/metrics/{baseline,distilbert}.json (acc, macro-F1, weighted-F1)
                      ├─► reports/metrics/{baseline,distilbert}_report.csv (per-class P, R, F1)
                      ├─► reports/figures/{baseline,distilbert}_confusion.png (row-normalized)
                      ├─► reports/figures/{baseline,distilbert}_per_class_f1.png
                      └─► reports/metrics/{baseline,distilbert}_top_errors.csv
                              │                                       │
                              ▼                                       ▼
====================================================    ============================================
           COMPARATIVE SYNTHESIS & AUDIT                     DYNAMIC PRESET POOL MANAGEMENT
                 [src/compare.py]                                   [src/presets.py]
                      │                                               │
                      ├─► comparison_table.csv & .md                  ├─► Filters distilbert_test_preds:
                      ├─► compare_per_class_f1.png                    │   true==pred, conf >= 0.90, 6-40 words
                      ├─► compare_overall.png                         ├─► FALLBACK_POOLS (15/emotion)
                      └─► error_analysis.md (3-way error taxonomy)    └─► next_sample(label, state):
                                                                          Shuffled non-repeating queue
                              │                                       │
                              └───────────────────┬───────────────────┘
                                                  │
                                                  ▼
====================================================================================================
                              EXPLAINABLE AI & INFERENCE RUNTIME
                                [src/explain.py & src/predict.py]
                      │
                      ├─► src/explain.py: LimeTextExplainer with batched perturbation inference
                      │   └─► reports/figures/lime_example_{1..5}.html & .png
                      │
                      └─► src/predict.py: Unified EmotionPredictor class
                          • Lazy model loading
                          • Empty string validation
                          • Truncation to 64 tokens
                          • Identical output schema: {label, confidence, all_probs}
                                              │
                                              ▼
====================================================================================================
                                INTERACTIVE STREAMLIT APPLICATION
                                            [app.py]
               (Cached resources via @st.cache_resource & @st.cache_data, 5 tabs)
  ┌───────────────┬───────────────┬────────────────┬───────────────┬────────────────┐
  ▼               ▼               ▼                ▼               ▼                ▼
[Sidebar]      [Tab 1: Analyze][Tab 2: Explain] [Tab 3: Compare] [Tab 4: Batch]  [Tab 5: Results]
Model selector 6 dynamic preset LIME token       Side-by-side     CSV upload with Live metrics,
& benchmark    buttons + queue  weights & hover  prediction &     progress bar &  plots & error
metrics        prob bar chart   HTML embedding   divergence alert CSV export      analysis
====================================================================================================
```

---

## 📂 Section 1: Source Modules (`src/`)

### 1. `src/config.py`
- **Purpose**: Acts as the authoritative single source of truth for all global configurations, reproducibility seeds, filesystem paths, and canonical emotion label mappings. It guarantees that no other script in the repository hardcodes directory locations, random states, or label indexing orders.
- **Main Functions & Constants**:
  - `set_seed(seed=SEED)`: Seeds Python's `random`, `numpy`, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`), and sets `os.environ["PYTHONHASHSEED"]` to enforce deterministic execution across runs.
  - `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR`, `PREDICTIONS_DIR`, `METRICS_DIR`: Absolute `pathlib.Path` objects pointing to project directories, automatically creating them at runtime if they do not exist.
  - `LABELS`: Canonical list of emotion labels in exact Hugging Face index order: `["sadness", "joy", "love", "anger", "fear", "surprise"]`.
  - `id2label` and `label2id`: Two-way lookup dictionaries converting integer class IDs to strings and vice versa.
  - `MODEL_NAME = "distilbert-base-uncased"` and `MAX_LENGTH = 64`: Model identifier and maximum token sequence length.
- **Key Design Decision & Why**: Using `pathlib.Path` relative to `Path(__file__).resolve().parent.parent` instead of hardcoded relative strings or environment variables. This prevents operating system path separator failures (macOS/Linux `/` vs. Windows `\`) and ensures the code executes identically whether run from the project root, a test runner, or an interactive subshell.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why is having an authoritative label list in `config.py` strictly necessary when Scikit-Learn pipelines already store class labels in `model.classes_`?"*
  - *Defense*: *"Scikit-Learn sorts string classes alphabetically by default (`['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']`), whereas PyTorch models and the Hugging Face dataset follow the original integer ordering (`0: sadness, 1: joy, ...`). If we relied on `model.classes_` without canonical reordering, probability arrays would be indexed differently across models, corrupting downstream comparisons, metric evaluations, and LIME explanations."*

---

### 2. `src/data.py`
- **Purpose**: Downloads and loads the `dair-ai/emotion` benchmark corpus from the Hugging Face Hub and structures it into train, validation, and test splits. It extracts raw text strings alongside corresponding integer and string emotion targets.
- **Main Functions**:
  - `load_emotion_dataset()`: Calls Hugging Face's `datasets.load_dataset("dair-ai/emotion")` to download and cache the dataset locally.
  - `get_splits()`: Iterates over the `'train'`, `'validation'`, and `'test'` partitions, mapping each integer label to its canonical name using `id2label`, and returns a tuple of three clean dictionaries: `(train_data, val_data, test_data)`.
- **Key Design Decision & Why**: Packaging the Hugging Face `DatasetDict` into standard native Python dictionaries containing lists of strings and integers (`text`, `label`, `label_id`). This completely decouples downstream components (such as scikit-learn baseline pipelines and unit tests) from Hugging Face library abstractions, allowing components to run independently without requiring active internet access once cached.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why do you return both `'label'` (string) and `'label_id'` (integer) in each split dictionary?"*
  - *Defense*: *"Different modeling paradigms expect different target types. Scikit-Learn's classical classification pipeline natively accepts string class names to generate human-readable classification reports, whereas PyTorch and the Hugging Face `Trainer` require integer label IDs (`0..5`) to compute Cross-Entropy loss tensors. Supplying both prevents repetitive, error-prone conversions across the codebase."*

---

### 3. `src/eda.py`
- **Purpose**: Performs exploratory data analysis to profile class balance and text length characteristics across all dataset partitions. It produces summary statistics and visual charts in `reports/metrics/` and `reports/figures/` that guide preprocessing and modeling choices.
- **Main Functions**:
  - `analyze_dataset_distributions()`: Computes cross-tabulation of emotion classes across splits (`class_distribution.csv`), generates a Seaborn training split class distribution bar plot (`class_distribution.png`), plots word count histograms across splits (`text_length_distribution.png`), and computes word count percentiles (mean, median, 95th percentile, max, min).
  - `main()`: CLI entry point that runs the analysis and prints formatted summary statistics to the console.
- **Key Design Decision & Why**: Computing explicit text length percentiles (finding a 95th percentile of 41 words and a maximum of 66 words) before modeling. This empirical finding justified setting `MAX_LENGTH = 64` in `config.py`, which accommodates virtually the entire corpus without truncation while avoiding the quadratic memory overhead of padding to BERT's default 512 tokens.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Your EDA shows that joy (33.8%) and sadness (29.0%) make up over 62% of the dataset, while surprise makes up only 3.6%. How did this finding directly influence your evaluation metrics?"*
  - *Defense*: *"Because of severe class imbalance, standard accuracy is misleading: a naive majority-class classifier predicting only joy or sadness could score over 60% accuracy while failing completely on minority emotions. This directly motivated our choice of Macro-Averaged F1 as our primary benchmark metric, as it weights all six classes equally regardless of frequency."*

---

### 4. `src/preprocess.py`
- **Purpose**: Provides two specialized text cleaning pipelines: `heavy_clean` for classical bag-of-words modeling and `minimal_clean` for transformer tokenization. It also implements automated NLTK corpus management and protected emotion negation rules.
- **Main Functions**:
  - `_ensure_nltk_data()` / `ensure_nltk_resources()`: Lazily verifies and downloads required NLTK tokenizers and lexicons (`stopwords`, `wordnet`, `omw-1.4`) if missing.
  - `_get_stopwords()`: Returns standard NLTK English stopwords minus protected negation terms.
  - `_get_lemmatizer()`: Returns a cached instance of `WordNetLemmatizer`.
  - `minimal_clean(text)`: Collapses irregular whitespace and strips URLs and user mentions while preserving casing, punctuation, and sentence grammar.
  - `heavy_clean(text, remove_stopwords=False, lemmatize=True)`: Lowercases text, standardizes apostrophes, expands contractions (e.g., `can't` $\rightarrow$ `can not`), removes non-alphanumeric punctuation, lemmatizes words, and preserves negations.
- **Key Design Decision & Why**: The **Negation Preservation Rule** (`NEGATION_WORDS = {"not", "no", "never", "n't", "nor"}`). Standard NLP stopword filters discard negation words, causing catastrophic sentiment inversion (e.g., *"not happy"* collapses to *"happy"*). By explicitly filtering out negations from the stopword set and expanding contractions, our classical pipeline retains affective valence.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why didn't you apply `heavy_clean` to DistilBERT as well so both models saw the exact same input?"*
  - *Defense*: *"Pre-trained transformers like DistilBERT were pre-trained on natural, unstripped text using WordPiece subwords. They rely heavily on punctuation marks, capitalization, and syntactic phrasing to calculate attention weights across tokens. Applying aggressive lemmatization and punctuation stripping destroys these structural signals, degrading transformer performance."*

---

### 5. `src/evaluate.py`
- **Purpose**: Serves as the centralized evaluation engine for both models, generating consistent performance metrics and standardized prediction files. It ensures an unbiased, apples-to-apples comparison between classical and deep learning approaches.
- **Main Functions**:
  - `save_predictions(model_name, texts, y_true, probs, output_dir)`: Exports test set predictions to a CSV file matching the project contract: `text`, `true_label`, `pred_label`, `confidence`, and the 6 class probability columns in canonical order.
  - `compute_metrics(y_true, y_pred, model_name, output_dir)`: Calculates accuracy, macro-F1, and weighted-F1; safely merges these values into the model's metrics JSON without overwriting existing benchmarking metadata.
  - `save_classification_report(model_name, y_true, y_pred, output_dir)`: Generates and exports Scikit-Learn's per-class precision, recall, and F1-score table as a CSV file.
  - `plot_confusion_matrix(model_name, y_true, y_pred, output_dir)`: Generates and saves a row-normalized $6 \times 6$ confusion matrix heatmap.
  - `plot_per_class_f1(model_name, y_true, y_pred, output_dir)`: Generates an annotated bar chart showing F1 scores for each emotion class.
  - `top_confident_errors(model_name, n=10, predictions_dir, output_dir)`: Identifies and exports the top $N$ highest-confidence false predictions to CSV for qualitative error analysis.
  - `evaluate_model(model_name, texts, y_true, probs, output_dirs)`: Orchestrates the entire evaluation workflow in a single function call.
- **Key Design Decision & Why**: Decoupling probability generation from metric evaluation by passing probability matrices (`probs`) directly to `evaluate_model`. This allows the evaluation module to remain completely agnostic to whether predictions were computed by Scikit-Learn or PyTorch, preventing code duplication across models.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why does your confusion matrix plot normalize by `true` (rows) rather than showing raw sample counts?"*
  - *Defense*: *"Because the test set has 695 joy samples and only 66 surprise samples, raw counts visually hide misclassifications on minority categories. Row normalization converts each cell into the proportion of actual class instances ($Recall_c$ on the diagonal), allowing us to immediately diagnose confusion patterns on minority classes regardless of class frequency."*

---

### 6. `src/baseline.py`
- **Purpose**: Implements the classical Machine Learning baseline using TF-IDF feature extraction and multinomial Logistic Regression. It handles automated hyperparameter tuning on the validation split, model training, artifact serialization, and test evaluation.
- **Main Functions**:
  - `build_pipeline(C=1.0, ngram_range=(1, 1))`: Constructs a Scikit-Learn `Pipeline` combining `TfidfVectorizer(max_features=20000)` and `LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')`.
  - `tune_baseline(X_train_clean, y_train, X_val_clean, y_val)`: Uses `GridSearchCV` with `PredefinedSplit` over the validation fold to evaluate $C \in \{0.5, 1.0, 5.0, 10.0\}$ and unigram vs. bigram feature sets.
  - `train_baseline(X_train_clean, y_train, best_params)`: Fits the optimal hyperparameter configuration ($C=5.0$, unigrams) strictly on the training set and logs training time.
  - `run_baseline()`: Top-level driver that coordinates data loading, cleaning, tuning, refitting, saving `models/baseline.joblib`, latency benchmarking, and evaluation artifact generation.
- **Key Design Decision & Why**: Using `PredefinedSplit` during grid search combined with `class_weight='balanced'`. `PredefinedSplit` guarantees that hyperparameter selection uses the exact official validation partition without data leakage across random folds. Setting balanced class weights automatically scales inverse penalties so minority classes are not ignored by the loss function.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why did you use Logistic Regression instead of a Support Vector Machine (SVM) or Naive Bayes for the classical baseline?"*
  - *Defense*: *"Logistic Regression with L-BFGS natively produces smooth, well-calibrated posterior class probabilities via the softmax function. In contrast, Naive Bayes makes unrealistic feature independence assumptions that produce overly confident, extreme probabilities, while standard SVMs require computationally expensive Platt scaling to generate probabilities."*

---

### 7. `src/transformer.py`
- **Purpose**: Fine-tunes the `distilbert-base-uncased` transformer architecture for multi-class sequence classification using the Hugging Face `Trainer` API. It handles tokenization, dynamic padding, GPU/CPU execution settings, training curve plotting, and full test evaluation.
- **Main Functions**:
  - `EmotionDataset`: Custom PyTorch `Dataset` wrapper storing tokenized tensor dictionaries (`input_ids`, `attention_mask`) and integer label targets.
  - `compute_eval_metrics(eval_pred)`: Evaluation callback invoked by the Trainer at each epoch to compute validation accuracy, macro-F1, and weighted-F1.
  - `get_dir_size_mb(path)`: Utility that calculates the total disk footprint of saved model artifacts in megabytes.
  - `plot_training_curves(log_history, output_path, model_name)`: Plots training vs. validation loss and validation accuracy vs. macro-F1 over epochs (`distilbert_training_curves.png`).
  - `train_transformer(subset=None, epochs=3, batch_size=32)`: Orchestrates the fine-tuning process, configures `TrainingArguments` with `load_best_model_at_end=True` monitored on validation `macro_f1`, saves the model to `models/distilbert/`, and measures inference latency.
- **Key Design Decision & Why**: Using `DataCollatorWithPadding` for dynamic padding combined with checkpoint selection based on validation `macro_f1`. Dynamic padding pads sequences only to the maximum length of the current mini-batch rather than the global maximum length (64 tokens), cutting padding computation by ~60%. Monitoring macro-F1 ensures the saved checkpoint prioritizes performance on minority classes.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why did you choose DistilBERT over larger architectures like BERT-base or RoBERTa?"*
  - *Defense*: *"DistilBERT uses knowledge distillation during pre-training to retain 97% of BERT's language understanding while reducing parameter count by 40% (66M parameters) and running 60% faster. This computational efficiency allowed us to perform full end-to-end training and sub-second inference entirely on commodity CPUs without requiring dedicated cloud GPUs."*

---

### 8. `src/compare.py`
- **Purpose**: Compares the classical baseline and DistilBERT models side-by-side across all evaluation metrics, inference speeds, and model footprints. It produces comparative synthesis tables, grouped visualization bar charts, and a detailed cross-model error analysis report.
- **Main Functions**:
  - `check_required_files(predictions_dir, metrics_dir)`: Verifies that required prediction CSVs and metrics JSON files exist for both models while ignoring smoke-test files.
  - `load_model_metrics(metrics_file, model_file_or_dir)`: Parses model metric JSONs and dynamically calculates disk footprint if missing.
  - `generate_comparison_table(...)`: Produces `comparison_table.csv` and `comparison_table.md` summarizing accuracy, F1 scores, latency, and model size.
  - `plot_grouped_per_class_f1(...)`: Generates a grouped bar chart (`compare_per_class_f1.png`) comparing per-class F1 scores.
  - `plot_grouped_overall_metrics(...)`: Generates a grouped bar chart (`compare_overall.png`) comparing overall accuracy, macro-F1, and weighted-F1.
  - `generate_error_analysis(...)`: Segments the 2,000 test predictions into three discrepancy categories (Baseline wrong / DistilBERT right, DistilBERT wrong / Baseline right, Both wrong) and exports `reports/metrics/error_analysis.md`.
  - `run_comparison(...)`: Top-level CLI and programmatic execution driver.
- **Key Design Decision & Why**: Creating an automated 3-way error discrepancy taxonomy rather than relying only on aggregate numbers. Categorizing instances where one model succeeds while the other fails reveals qualitative insights—such as showing where DistilBERT resolves complex syntactic structures and where simple keyword matching still performs well.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"What does your comparison analysis reveal about the trade-off between the Baseline and DistilBERT?"*
  - *Defense*: *"DistilBERT achieves higher accuracy (+4.70 percentage points to 92.95%) and macro-F1 (+4.90 percentage points to 88.32%), especially on minority classes like love (+6.49%) and surprise (+5.65%). However, the baseline model trains 530× faster (0.87s vs. 460.59s), runs inference 535× faster (5.09 ms vs. 2,725.15 ms per 1,000 samples), and is 233× smaller on disk (1.10 MB vs. 256.12 MB), making it ideal for resource-constrained edge deployments."*

---

### 9. `src/explain.py`
- **Purpose**: Integrates Local Interpretable Model-agnostic Explanations (LIME) to explain DistilBERT predictions at the token level. It quantifies positive and negative word contributions and exports interactive HTML and PNG visualizations.
- **Main Functions**:
  - `get_model_and_tokenizer(model_dir)`: Loads and caches the DistilBERT model and tokenizer singleton.
  - `get_explainer()`: Initializes and caches the `LimeTextExplainer(class_names=LABELS, bow=False)` instance.
  - `predict_proba(texts, batch_size=32)`: Computes batched softmax probabilities for perturbed texts generated by LIME, with safety checks for empty strings.
  - `explain_text(text, num_samples=500, num_features=8)`: Fits a local linear surrogate model and returns the predicted label, class probabilities, and a list of `(word, weight)` tuples.
  - `save_explanation(text_or_explanation, path)`: Saves the interactive explanation as a standalone HTML file.
  - `plot_explanation_barchart(weights, predicted_label, confidence, output_path)`: Renders a horizontal bar chart of feature weights with green (positive) and red (negative) bars.
  - `run_explanations_demo()`: Generates 5 benchmark HTML and PNG explanations in `reports/figures/`, including a real misclassification from `distilbert_top_errors.csv`.
- **Key Design Decision & Why**: Implementing empty-string fallback protection inside `predict_proba`. LIME perturbs sentences by randomly masking out words; on short texts, perturbations can remove all words, creating empty strings. Standard tokenizers crash on empty strings; our implementation replaces them with safe whitespace tokens to prevent crashes during sampling.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"How does LIME compute token weights for a deep neural network like DistilBERT?"*
  - *Defense*: *"LIME is model-agnostic: it draws $N$ perturbed samples by randomly masking words from the original text, passes them through DistilBERT to get probability outputs, weights each sample by its exponential cosine distance to the original input, and fits a sparse linear regression model. The regression coefficients represent the local feature weights of each word toward the predicted class."*

---

### 10. `src/predict.py`
- **Purpose**: Provides a unified, object-oriented inference API (`EmotionPredictor`) and command-line interface for classifying raw text inputs. It abstracts away model-specific preprocessing, handles truncation, and guarantees an identical output format across both architectures.
- **Main Functions**:
  - `EmotionPredictor.__init__(model_type)`: Validates that `model_type` is `'baseline'` or `'distilbert'` and initializes internal cache references.
  - `EmotionPredictor._ensure_loaded()`: Lazily loads either `models/baseline.joblib` or `models/distilbert/` on first inference, raising clear `FileNotFoundError` instructions if weights are missing.
  - `EmotionPredictor.predict(text)`: Validates that the input is non-empty and non-whitespace, calls `predict_batch([text])`, and returns a single prediction dictionary.
  - `EmotionPredictor.predict_batch(texts)`: Validates all texts, runs model-appropriate preprocessing (`heavy_clean` or `minimal_clean`), tokenizes/vectorizes, truncates long texts to 64 tokens, aligns probabilities to canonical label order, and returns structured dictionaries.
  - `main()` / `parse_args()`: CLI interface allowing commands like `python -m src.predict "text" --model distilbert`.
- **Key Design Decision & Why**: **Lazy loading** coupled with a **Polymorphic Output Contract**. Model weights are loaded into memory only when the first prediction is requested, avoiding startup delays when importing the module. Both models return the exact same dictionary structure (`label`, `confidence`, `all_probs`), allowing frontend applications to switch between models without changing application logic.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why does `EmotionPredictor.predict()` raise a `ValueError` on empty strings while `src/preprocess.py` returns an empty string?"*
  - *Defense*: *"This follows the principle of separation of concerns. Preprocessing functions are data cleaners designed to handle dirty inputs gracefully without throwing exceptions. The inference API, however, is a user-facing contract where classifying an empty string is an invalid semantic request that should fail explicitly rather than producing arbitrary predictions."*

---

### 11. `src/presets.py`
- **Purpose**: Manages curated sentence pools for quick preset buttons in the web application, extracting high-confidence predictions from the test set with rich hand-written fallbacks. It provides stateful queue sampling so users never receive immediate duplicate sentences when cycling through emotion examples.
- **Main Functions & Constants**:
  - `FALLBACK_POOLS`: Constant dictionary containing at least 15 clearly emotional, hand-written sentences for each of the 6 canonical emotions (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`) to guarantee system availability if test prediction files are missing or underpopulated.
  - `build_pools(csv_path=None)`: Reads `distilbert_test_preds.csv`, keeps instances where `true_label == pred_label`, `confidence >= 0.90`, and word count is between 6 and 40 words, drops duplicates, and supplements with `FALLBACK_POOLS` if any class has fewer than 15 sentences.
  - `next_sample(label, state, pools=None)`: Manages a shuffled queue stored in `state[f"queue_{label}"]`. When empty, it rebuilds and shuffles the full pool, ensuring that the first sentence of the new cycle does not equal the last sentence shown, and pops the next sample using a time-seeded random generator.
- **Key Design Decision & Why**: Using a stateful shuffled queue that swaps the first element if it matches `last_shown` across queue regenerations. This ensures two crucial properties: within a cycle of size $N$, all draws are unique without repeats, and at cycle boundaries, immediate consecutive repetition is strictly prevented, all while seeding from system time (`time.time_ns()`) so every session experiences a fresh, non-deterministic demonstration order.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why do you seed `random.Random(time.time_ns())` inside `next_sample` instead of using the global `SEED = 42` from `config.py`?"*
  - *Defense*: *"Global reproducibility seeds like `SEED = 42` are essential for model training and evaluation benchmarks so test metrics are deterministic. However, for interactive UI presets, using a fixed seed would force every browser session to present the exact same sequence of sentences in the exact same order. Seeding from system time ensures an organic, non-repeating exploration experience for users while preserving deterministic guarantees in the underlying models."*

---

## 🖥️ Section 2: Interactive Web Application (`app.py`)

### 12. `app.py`
- **Purpose**: Implements an interactive 5-tab Streamlit web application providing a user interface for emotion classification, LIME explainability, model comparison, batch CSV processing, and evaluation dashboard inspection. It serves as the primary demonstration portal for end users.
- **Main Functions**:
  - `get_cached_predictor(model_type)`: Decorated with `@st.cache_resource` to load and persist `EmotionPredictor` instances in memory across browser interactions.
  - `get_cached_pools()`: Decorated with `@st.cache_data` to construct and cache the high-confidence preset sentence pools from `src.presets.build_pools()`.
  - `safe_get_predictor(model_type)`: Wraps predictor retrieval in user-friendly error banners if model checkpoints have not been trained yet.
  - `render_analyze_tab(selected_model_type)`: Tab 1 — Interactive single-text classification with 6 dynamic preset emotion buttons (`Joy`, `Sadness`, `Love`, `Anger`, `Fear`, `Surprise`) using `on_click=set_text` calling `next_sample` to cycle through non-repeating samples into `st.session_state["input_text"]`, followed by `st.text_area(key="input_text")` without `value=`, confidence badge, and posterior probability bar chart.
  - `render_explain_tab(selected_model_type)`: Tab 2 — In-situ LIME visualization embedding HTML via `streamlit.components.v1.html` for token-level explainability, reading active text from `st.session_state.get("input_text", ...)`.
  - `render_compare_tab()`: Tab 3 — Side-by-side inference comparing Baseline and DistilBERT on identical input with agreement/disagreement callouts, reading active text from `st.session_state.get("input_text", ...)`.
  - `render_batch_tab(selected_model_type)`: Tab 4 — CSV upload validation (`text` column check), progress bar batched inference, and results CSV export.
  - `render_results_tab()`: Tab 5 — Dashboard displaying benchmark tables, confusion matrices, and training curves directly from `reports/`.
  - `main()`: Configures page layout, injects custom CSS styling, renders the sidebar, and coordinates tab navigation.
- **Key Design Decision & Why**: Using Streamlit's `on_click` callback combined with `st.session_state["input_text"]` on `st.text_area` without a `value=` parameter. In Streamlit's execution model, widget callbacks run *before* the script body executes. Defining preset buttons with `on_click` updates the session state before the text area is instantiated, guaranteeing immediate text replacement without Streamlit warnings or input synchronization lag.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why did you use `on_click` callbacks and `st.session_state['input_text']` instead of just checking `if st.button('Joy'): user_text = ...`?"*
  - *Defense*: *"In Streamlit, if you check `if st.button():` and update a text area that defines a `value` argument alongside a session `key`, Streamlit throws a deprecation warning or fails to synchronize user keystrokes with button clicks. By using `on_click` and letting `st.text_area(key='input_text')` bind directly to session state without `value=`, widget state changes execute atomically before DOM rendering, ensuring seamless, bug-free bidirectional updates."*

---

## 🧪 Section 3: Test Suite Modules (`tests/`)

### 13. `tests/test_preprocess.py`
- **Purpose**: Unit test suite that validates the text preprocessing functions in `src/preprocess.py`. It guarantees that edge-case inputs, contraction expansions, lemmatization, and negation words are handled correctly.
- **Main Functions**:
  - `test_empty_and_none_input()`: Asserts that passing `None`, `""`, or pure whitespace to `minimal_clean` and `heavy_clean` returns an empty string `""` without throwing exceptions.
  - `test_minimal_clean_preserves_structure()`: Verifies that URLs and mentions are removed while uppercase casing, punctuation, and contractions remain intact.
  - `test_heavy_clean_contractions_and_lemmatization()`: Verifies contraction expansion (e.g. `shouldn't` $\rightarrow$ `should not`), punctuation stripping, and noun lemmatization (e.g. `apples` $\rightarrow$ `apple`).
  - `test_negations_preserved_with_stopword_removal()`: Asserts that critical negation words (`not`, `never`, `nor`, `no`) are never stripped even when `remove_stopwords=True`.
- **Key Design Decision & Why**: Dedicated negation preservation test asserting that non-negation words (like *"am"*, *"will"*, *"have"*) are stripped while negations are preserved. This prevents silent regressions that would invert affective polarity.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why is testing empty and `None` inputs critical for preprocessing routines in an NLP system?"*
  - *Defense*: *"Real-world user input and web scrapers frequently encounter null values, empty form submissions, or strings composed solely of whitespace or URLs that become empty after regex stripping. If the preprocessor threw an unhandled TypeError or AttributeError, it would cause server crashes in production APIs and the Streamlit app."*

---

### 14. `tests/test_evaluate.py`
- **Purpose**: Unit test suite that validates metric computation, classification report generation, and artifact export in `src/evaluate.py`. It verifies that the evaluation pipeline adheres strictly to the project's prediction file contract.
- **Main Functions**:
  - `test_evaluate_model_pipeline_with_dummy_data(tmp_path)`: Uses isolated temporary directories and synthetic 6-class probability arrays with an intentional error to verify that predictions CSVs, JSON metrics, classification reports, confusion heatmaps, per-class F1 charts, and top error files are correctly generated.
  - `test_compute_metrics_merges_without_overwriting(tmp_path)`: Verifies that calling `compute_metrics` merges new metrics (`accuracy`, `macro_f1`, `weighted_f1`) into existing JSON files without deleting pre-existing metadata such as inference time or device info.
- **Key Design Decision & Why**: Using pytest's `tmp_path` fixture to isolate all filesystem side-effects. This prevents unit tests from modifying, overwriting, or corrupting production benchmark files in `reports/metrics/`.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"How does `test_evaluate_model_pipeline_with_dummy_data` verify that your error analysis function works properly?"*
  - *Defense*: *"The dummy probability matrix deliberately introduces a high-confidence misclassification on sample 4 (true label `'fear'` predicted as `'anger'` with confidence 0.75). The test asserts that `dummy_test_model_top_errors.csv` contains exactly 1 error row and that the confidence, true label, and predicted label match the synthetic error."*

---

### 15. `tests/test_predict.py`
- **Purpose**: Unit test suite verifying the contract, error handling, truncation, and output format consistency of `EmotionPredictor` in `src/predict.py`. It ensures that both baseline and DistilBERT models conform to identical API expectations.
- **Main Functions**:
  - `test_empty_string_raises_value_error()`: Asserts that empty strings, whitespace, and batches containing blank items raise `ValueError` with clear messages.
  - `test_long_string_truncation()`: Asserts that a 560-word input is safely truncated to 64 tokens without throwing index or dimension mismatch errors.
  - `test_batch_size_matches_input()`: Verifies that passing $N$ texts to `predict_batch` returns a list of length $N$.
  - `test_predicted_label_is_valid()`: Asserts that predictions belong to canonical `config.LABELS`.
  - `test_probabilities_sum_to_one()`: Verifies that posterior class probabilities sum to 1.0 within floating-point tolerance.
  - `test_both_models_give_same_output_format()`: Asserts that baseline and DistilBERT return identical dictionary schemas and keys.
- **Key Design Decision & Why**: Parameterizing tests across `["baseline", "distilbert"]`. Testing both models against the exact same test assertions guarantees polymorphic interchangeability: code interacting with `EmotionPredictor` does not need to know which model is active.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"What guarantees does `test_probabilities_sum_to_one()` provide for downstream consumers?"*
  - *Defense*: *"It verifies that probability distributions output by both the scikit-learn baseline and the PyTorch transformer are properly normalized valid probability measures summing to 1.0. This prevents rendering skewed or invalid confidence bars in the Streamlit UI."*

---

### 16. `tests/test_explain.py`
- **Purpose**: Unit test suite that verifies the robustness and functionality of the LIME explainability module in `src/explain.py`. It validates empty-string handling, probability sum constraints, and HTML/PNG export routines.
- **Main Functions**:
  - `test_empty_or_whitespace_input_returns_error()`: Verifies that passing `None`, `""`, or whitespace returns a graceful dictionary containing an error message rather than crashing.
  - `test_predict_proba_handles_empty_strings()`: Verifies that batched probability inference safely evaluates batches containing empty strings generated during LIME word-masking.
  - `test_explain_text_and_saving(tmp_path)`: Runs LIME with a small sample budget (`num_samples=50`) to verify weight generation, interactive HTML file creation, and bar chart PNG rendering.
- **Key Design Decision & Why**: Using a small sample budget (`num_samples=50`) during unit tests instead of the production default (`500`). This ensures the test suite executes in under 8 seconds while fully testing the LIME perturbation and visualization code path.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why is `test_predict_proba_handles_empty_strings` essential specifically for LIME?"*
  - *Defense*: *"LIME's perturbation mechanism randomly replaces words with masks to build sample neighborhoods. In short sentences, some perturbations mask out every single word, generating empty strings; if `predict_proba` crashed on empty strings, LIME would fail midway through its sampling loop."*

---

### 17. `tests/test_compare.py`
- **Purpose**: Unit test suite verifying the comparative evaluation and synthesis engine in `src/compare.py`. It validates file dependency checks, table formatting, grouped chart generation, and error segmentation.
- **Main Functions**:
  - `test_check_required_files_detects_missing(tmp_path)`: Verifies that missing prediction CSVs or metrics JSONs are correctly detected and reported.
  - `test_compare_generates_all_artifacts(tmp_path)`: Uses synthetic prediction data across 12 samples to test that `run_comparison` generates `comparison_table.csv`, `comparison_table.md`, `compare_per_class_f1.png`, `compare_overall.png`, and `error_analysis.md`.
- **Key Design Decision & Why**: Constructing a synthetic 12-sample test dataset that explicitly includes all three error taxonomy cases (baseline wrong/DistilBERT right, DistilBERT wrong/baseline right, and both wrong). This thoroughly verifies error categorization logic without requiring long model training runs.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"How does `check_required_files` protect against publishing preliminary test results?"*
  - *Defense*: *"It strictly filters out any file names matching `"*_smoke*"`, ensuring that quick debugging or smoke-test runs (e.g. 500 samples, 1 epoch) are never accidentally merged into the benchmark comparison tables and final reports."*

---

### 18. `tests/test_presets.py`
- **Purpose**: Unit test suite that validates the sentence pool generation and stateful sampling queue logic in `src/presets.py`. It guarantees non-repetition across cycles, draw uniqueness within cycles, string validity across all classes, and robust fallback behavior when predictions are absent.
- **Main Functions**:
  - `test_no_immediate_repetition()`: Executes 200 consecutive `next_sample` draws for `'joy'` and asserts that no two adjacent draws are identical ($x_t \neq x_{t-1}$).
  - `test_unique_draws_until_exhaustion()`: Asserts that for every emotion class, the first $\min(\text{pool\_size}, 50)$ draws contain zero duplicates, verifying strict cycle uniqueness.
  - `test_every_label_returns_non_empty_string()`: Verifies that every canonical emotion returns a non-empty string with between 6 and 40 words.
  - `test_fallback_when_csv_missing(tmp_path)`: Uses a non-existent CSV path to verify that `build_pools` gracefully falls back to `FALLBACK_POOLS` with $\ge 15$ sentences per class, and tests that 100 consecutive draws on the fallback pool also never repeat consecutively.
- **Key Design Decision & Why**: Testing cycle transitions by executing 200 consecutive draws on both large prediction pools and small 15-sample fallback pools. With a pool of 15, the queue resets more than 13 times in 200 draws, thoroughly stressing the boundary swap logic where cycle $k$ transitions to cycle $k+1$.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why does `test_unique_draws_until_exhaustion` check `min(pool_size, 50)` instead of checking a fixed 50 draws for all classes?"*
  - *Defense*: *"Because the underlying benchmark dataset is severely imbalanced, the high-confidence surprise pool contains 31 samples, which is fewer than 50. Drawing 50 samples would require starting a second cycle, which by definition must repeat samples from the first cycle. Testing `min(pool_size, 50)` strictly verifies that all draws within a single complete cycle are unique without conflating cycle exhaustion with duplicate bugs."*

---

### 19. `src/data_v2.py`
- **Purpose**: Multi-source dataset engineering module that constructs `data/train_v2.csv` (17,992 samples) by merging `dair-ai/emotion` with Google's `GoEmotions`, removing repetitive `"i feel"` templates, and enforcing strict zero-leakage protection.
- **Main Functions**:
  - `rephrase_prefix(text, rng)`: Strips or naturally varies leading emotional prefixes (`"i feel"`, `"i am feeling"`).
  - `build_train_v2()`: Ingests original training data, filters single-label GoEmotions examples, projects 27 labels to 6 canonical labels via `GOEMOTIONS_MAP`, deduplicates, and filters against `val.csv`, `test.csv`, and `probe_set.csv`.
- **Key Design Decision & Why**: Projecting granular GoEmotions classes (e.g. *admiration*, *caring* $\rightarrow$ `love`; *annoyance* $\rightarrow$ `anger`) while strictly dropping ambiguous multi-label examples. This provided 1,992 diverse external samples without diluting label precision.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"How did you prevent data leakage when creating `train_v2.csv`?"*
  - *Defense*: *"Before inserting any external or augmented sample, we collected all lowercase, stripped text strings from `val.csv`, `test.csv`, and `probe_set.csv` into a hash set. Any candidate sample matching this set was immediately discarded, guaranteeing zero test or probe contamination."*

---

### 20. `src/calibrate.py`
- **Purpose**: Model calibration and selective prediction module. Fits a post-hoc temperature scaling parameter $T$ on validation logits via NLL minimization, computes Expected Calibration Error (ECE) across 10 bins, and chooses an optimal confidence abstain threshold on the probe dev split.
- **Main Functions**:
  - `fit_temperature(val_logits, val_labels)`: Solves $\arg\min_T \text{NLL}(z / T, y)$ using `scipy.optimize.minimize_scalar`.
  - `compute_ece(logits, labels, n_bins=10)`: Computes empirical binned calibration error $\sum_m \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
  - `choose_abstain_threshold(calibrated_probs, label_names)`: Scans thresholds 0.40 to 0.90 on probe `dev` to maximize accuracy on answered items subject to $\ge 70\%$ coverage while rejecting neutral out-of-domain rows.
  - `plot_reliability_diagram()`: Generates side-by-side reliability diagrams before and after temperature scaling.
- **Key Design Decision & Why**: Using **only** the `dev` partition of `data/probe_set.csv` to select the abstain threshold ($\tau = 0.90$), keeping the probe `test` split completely untouched until final promotion evaluation.
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"Why does temperature scaling improve calibration without altering accuracy?"*
  - *Defense*: *"Because temperature scaling divides all logits by a single positive scalar $T$, it is a strictly monotonic transformation. The $\arg\max$ of the softmax output is unchanged, so predictions and top-1 accuracy remain identical, but overconfident probabilities are softened to match true empirical accuracies."*

---

### 21. `src/probe_eval.py`
- **Purpose**: Independent evaluation module benchmarking models on `data/probe_set.csv` (144 sentences: 60 emotional + 12 neutral per split), simulating realistic faculty evaluation and evaluating out-of-domain neutral false-positive rates.
- **Main Functions**:
  - `evaluate_probe(model_type, split)`: Computes accuracy, macro-F1, per-class F1, confident-only accuracy, and falsely confident neutral rates.
- **Key Design Decision & Why**: Separating emotional sentences (used for standard accuracy and macro-F1) from neutral sentences (used for OOD false confidence testing).
- **Faculty Oral Exam Question & Defense**:
  - *Question*: *"What constitutes a 'falsely confident neutral prediction'?"*
  - *Defense*: *"Neutral sentences contain no valid emotion. If a model predicts any emotion with confidence $\ge 0.80$ (or above the abstain threshold), it is falsely confident. A robust model should output low confidence or abstain."*

---

### 22. `tests/test_data_v2.py` & `tests/test_probe_set.py`
- **Purpose**: Automated regression tests validating dataset schemas, split balances, sentence word count bounds (5-25 words), label mappings, and strict zero data leakage.
- **Main Functions**:
  - `test_no_overlap_with_val_test_probe()`: Verifies zero text overlap between `train_v2.csv` and all evaluation splits.
  - `test_splits_are_balanced()`: Confirms exact 10-sample balance per emotion class across dev and test probe partitions.
