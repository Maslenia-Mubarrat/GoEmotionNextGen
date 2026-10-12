# EmoGenX: Augmented Multi-Label Emotion Classification for Long-Tail GoEmotions

EmoGenX is a research and experimentation pipeline for multi-label emotion classification on the GoEmotions dataset, designed to address class imbalance in minority emotions. The project combines data preparation, LLM-based synthetic augmentation, teacher-guided filtering, and a hybrid model architecture that blends contextual embeddings with label-relationship reasoning for stronger performance on rare emotion classes.

The repository is organized around a clear experimental workflow: prepare data, train baselines, augment minority labels, validate synthetic quality, train the core model, and evaluate ablations, threshold calibration, and explainability.

![EmoGenX Architecture](./EmoGenX%20architecture.svg)

## Why this project matters

Traditional emotion classification models often perform poorly on minority emotion labels because the dataset is highly imbalanced. EmoGenX tackles this problem in a data-engineering-friendly way:

- Converts the GoEmotions dataset into multi-hot label parquet datasets.
- Detects minority classes based on class frequency thresholds.
- Uses a large language model to generate synthetic examples for underrepresented emotions.
- Filters synthetic data using a teacher model confidence gate to retain only high-quality samples.
- Trains a custom model that combines a DeBERTa encoder with graph-like label dependency reasoning and contrastive learning.
- Evaluates fidelity and explainability using diversity metrics, t-SNE visualizations, threshold tuning, and attribution-based analysis.

This makes the project highly relevant for data science, ML engineering, and applied NLP work focused on data quality, class imbalance, and trustworthy model behavior.

## Core idea

The workflow follows a practical ML pipeline pattern:

1. Data ingestion and standardization
2. Baseline model training
3. Minority-class augmentation using LLM-generated text
4. Teacher-based filtering and synthetic-data validation
5. Training the final EmoGenX architecture
6. Evaluation via ablations and class-level threshold tuning
7. Explainability and faithfulness checks

## Project structure

```text
.
├── EmoGenX architecture.svg
├── EmoGenX architecture.zip
├── exp_01_data_prep.py
├── exp_02_baselines_training.py
├── exp_03_augmentation_baselines.py
├── exp_04_data_validation_fidelity.py
├── exp_05_train_emogenx_core.py
├── exp_06_ablation_study.py
├── exp_07_threshold_tuning_and_breakdown.py
├── exp_08_xai_faithfulness_and_cases.py
├── emogenx_artifacts/
│   ├── goemotions_train.parquet
│   ├── goemotions_val.parquet
│   ├── goemotions_test.parquet
│   ├── augmented_train_llama_filtered.parquet
│   ├── label_names.json
│   ├── minority_classes.json
│   ├── raw_synthetic_samples.json
│   ├── filtered_synthetic_samples.json
│   ├── baseline_deberta_best.pt
│   ├── emogenx_final_model.pt
│   ├── adj_matrix.npy
│   ├── table1_baselines_results.json
│   ├── table2_ablation_results.json
│   ├── table_diversity_metrics.json
│   ├── optimal_thresholds.json
│   └── figures/
└── README.md
```

## Repository goals

- Address class imbalance in multi-label emotion datasets
- Improve minority-class performance without relying only on raw oversampling
- Build a reproducible research pipeline with artifact tracking
- Make synthetic data quality measurable and explainable
- Showcase a practical end-to-end ML system for emotion recognition

## Stack

- Python
- PyTorch
- Hugging Face Transformers
- Datasets library
- Pandas / NumPy
- scikit-learn
- sentence-transformers
- Matplotlib
- NLTK

## Pipeline overview

### 1) Data preparation
`exp_01_data_prep.py`

- Loads the GoEmotions dataset from Hugging Face
- Converts label lists into a multi-hot representation
- Saves train, validation, and test parquet files
- Detects minority classes by frequency threshold
- Saves label metadata for downstream training and augmentation

### 2) Baselines
`exp_02_baselines_training.py`

- Trains a classical TF-IDF + Logistic Regression One-vs-Rest baseline
- Trains a DeBERTa-v3 multi-label classifier as the teacher model
- Saves the best teacher checkpoint and benchmark metrics

### 3) Synthetic augmentation for minority classes
`exp_03_augmentation_baselines.py`

- Selects minority labels such as grief, shame, or other long-tail emotions
- Generates realistic Reddit-style comments with an LLM prompt
- Builds synthetic training rows with label assignments
- Uses a teacher confidence gate to retain only high-confidence synthetic samples
- Produces unfiltered and filtered augmented datasets

### 4) Fidelity and validation
`exp_04_data_validation_fidelity.py`

- Calculates diversity metrics like Distinct-1, Distinct-2, and Self-BLEU
- Runs embedding-based t-SNE visual analysis to compare real vs synthetic minority samples
- Saves validation plots and metrics for auditability

### 5) Core model training
`exp_05_train_emogenx_core.py`

- Builds the EmoGenX model architecture
- Uses a DeBERTa encoder to produce contextual representations
- Creates a label dependency branch using a graph-like adjacency matrix
- Applies contrastive representation learning and Asymmetric Loss (ASL)
- Saves final model weights and label relation matrix

### 6) Ablation and threshold tuning
`exp_06_ablation_study.py`

- Compares the proposed EmoGenX architecture against baseline variants
- Saves ablation results for comparison

`exp_07_threshold_tuning_and_breakdown.py`

- Calibrates per-label operating thresholds using validation data
- Produces per-class F1 profile plots
- Optimizes model decision thresholds for multi-label evaluation

### 7) Explainability and faithfulness
`exp_08_xai_faithfulness_and_cases.py`

- Uses Integrated Gradients to compute model importance scores
- Checks token removal comprehensiveness
- Records explanation-focused metrics for interpretability analysis

## Getting started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended for training, though CPU execution is possible for smaller runs
- A working Python environment with pip

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets sentence-transformers scikit-learn pandas numpy matplotlib nltk
```

If needed, adjust the PyTorch installation to the correct CUDA version for your machine.

## Reproducible workflow

Run the scripts in order:

```bash
python exp_01_data_prep.py
python exp_02_baselines_training.py
python exp_03_augmentation_baselines.py
python exp_04_data_validation_fidelity.py
python exp_05_train_emogenx_core.py
python exp_06_ablation_study.py
python exp_07_threshold_tuning_and_breakdown.py
python exp_08_xai_faithfulness_and_cases.py
```

The outputs are stored in `./emogenx_artifacts` and include model weights, parquet datasets, JSON metrics, and figures.

## Output artifacts

The code automatically produces the following artifacts:

- `goemotions_train.parquet`
- `goemotions_val.parquet`
- `goemotions_test.parquet`
- `augmented_train_unfiltered.parquet`
- `augmented_train_llama_filtered.parquet`
- `label_names.json`
- `minority_classes.json`
- `raw_synthetic_samples.json`
- `filtered_synthetic_samples.json`
- `baseline_deberta_best.pt`
- `emogenx_final_model.pt`
- `adj_matrix.npy`
- `table1_baselines_results.json`
- `table2_ablation_results.json`
- `table_diversity_metrics.json`
- `optimal_thresholds.json`
- `table_xai_faithfulness_metrics.json`
- `figures/fig_tsne_distribution.png`
- `figures/fig_per_class_f1.png`

## Data engineering notes

This repository is not only a model experiment; it is also a strong example of data-centric ML engineering:

- Data artifacts are persisted in a machine-readable and version-friendly format (Parquet)
- Synthetic generation is controlled with a teacher-confidence gate instead of blind augmentation
- Quality checks are built into the flow via diversity and explainability metrics
- Model and dataset outputs are traceable and saved for downstream experimentation
- Experiment scripts are explicitly separated by stage, making the pipeline easy to maintain and extend

## Why a data-engineering hire would be impressed

A hiring manager or data engineering reviewer will notice that this project demonstrates:

- End-to-end ML pipeline design
- Dataset curation and minority-class handling
- Synthetic data generation and QA
- Reproducible experiment structure
- Artifact generation and validation
- Multi-stage model evaluation and explainability
- Real-world understanding of class imbalance and trustworthy ML workflows

This is a strong example of applied MLOps and ML engineering thinking in a research setting.

## Potential next steps

To make this production-ready, the project could be extended with:

- Dockerfiles and containerized training jobs
- CI/CD validation for dataset integrity and model smoke tests
- DVC or MLflow tracking for experiment lineage
- Cloud storage integration for artifact persistence
- A deployment API for inference and monitoring
- More robust prompt management and model governance for synthetic data generation

## License

This repository does not currently include a license file. If this project is meant for public or commercial use, add a proper license such as MIT or Apache-2.0 before publishing or sharing widely.

## Contact / maintainer

For questions about the project, model design, or data pipeline, contact the repository owner or maintainers through the GitHub project page.
