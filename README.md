# Lending AI — Fine-Tuned SLM for Credit Intelligence

Fine-tuning `meta-llama/Llama-3.2-3B-Instruct` with QLoRA on a lending portfolio dataset
to build a domain-aware Lending Intelligence Assistant.

## What the model does

| Task | Input | Output |
|------|-------|--------|
| **Loan Summary Generation** | Borrower profile + loan details | Natural language credit narrative with key risk signals |
| **Credit Risk Classification** | Bureau score, DPD, FOIR, history | Low / Medium / High Risk + chain-of-thought reasoning |
| **Loan Approval Recommendation** | Full applicant profile | Approve / Approve with Conditions / Reject + rationale |

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Place raw XLSX here (see setup below)
│   └── processed/            # Generated: train/val/test JSONL files
├── notebooks/
│   ├── 01_data_preparation.ipynb   # Dataset audit, cleaning, prompt engineering
│   ├── 02_fine_tuning.ipynb        # QLoRA training walkthrough
│   └── 03_evaluation.ipynb         # Before-vs-after comparison & business impact
├── src/
│   ├── data_prep.py          # Full data pipeline (run this first)
│   ├── train.py              # QLoRA fine-tuning script
│   └── evaluate.py           # Evaluation & report generation
├── configs/
│   └── training_config.yaml  # All hyperparameters, documented
├── outputs/
│   ├── adapter/              # Saved LoRA adapter weights (after training)
│   ├── training_metrics.json
│   ├── evaluation_report.md  # Human-readable before-vs-after report
│   └── evaluation_results.json
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Place `Lending_Loan_Portfolio_1000_Raw.xlsx` in `data/raw/`.

### 3. Accept the Llama model license

The model requires accepting Meta's license on HuggingFace:
1. Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Accept the license agreement
3. Set your HuggingFace token:

```bash
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

**Fallback** (no license required): Edit `configs/training_config.yaml` and set
`model.name: microsoft/phi-2`.

---

## Running the Pipeline

### Step 1: Data Preparation (run once)

```bash
python src/data_prep.py
```

Outputs:
- `data/processed/train.jsonl` — 2,400 training examples
- `data/processed/val.jsonl` — 300 validation examples
- `data/processed/test.jsonl` — 300 held-out test examples
- `data/processed/lending_cleaned.csv` — cleaned dataset

### Step 2: Fine-Tuning

```bash
python src/train.py
```

Training completes in **15–25 minutes** on a 12 GB GPU (or ~45 min on 8 GB GPU).

Outputs:
- `outputs/adapter/` — LoRA adapter weights
- `outputs/training_metrics.json` — loss curves and training stats

### Step 3: Evaluation

```bash
python src/evaluate.py
```

Outputs:
- `outputs/evaluation_report.md` — full before-vs-after comparison
- `outputs/evaluation_results.json` — structured metrics

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12–16 GB |
| System RAM | 16 GB | 32 GB |
| Disk space | 8 GB | 16 GB |

Peak VRAM during training: **~6.5–8 GB**

---

## Key Design Decisions

### Model: Llama-3.2-3B-Instruct

Selected because it (1) uses the full 3B parameter budget, (2) is already instruction-tuned
so LoRA teaches domain knowledge rather than also teaching instruction-following, and (3) has
native chat template support for SFTTrainer.

### Chain-of-thought completions

Completions include explicit reasoning steps before the final label. The model learns *why*
a borrower is High Risk (citing FOIR thresholds, Bureau Score tiers, DPD levels) — not just
*that* they are. This is measurably more useful to underwriters.

### Data cleaning rationale

Documented in `notebooks/01_data_preparation.ipynb`. Summary:
- LOAN_PRODUCT: 22 variants → 5 canonical via explicit mapping dict
- OCCUPATION: synonyms + nulls → 4 classes + 'Unknown'
- CURRENT_DPD: negative values clipped to 0; sentinel values (999) clipped to 365
- MONTHLY_INCOME nulls: imputed via (OCCUPATION × LOAN_PRODUCT) group median
- BUREAU_SCORE nulls: imputed via (LOAN_PRODUCT × COLLECTION_BUCKET) group median

---

## Submission Checklist

- [x] JSONL instruction-tuning dataset (`data/processed/`)
- [x] Fine-tuning script (`src/train.py`) and notebook (`notebooks/02_fine_tuning.ipynb`)
- [x] Evaluation report (`outputs/evaluation_report.md`)
- [x] Data preparation pipeline (`src/data_prep.py`, `notebooks/01_data_preparation.ipynb`)
- [x] Configuration files (`configs/training_config.yaml`)
- [x] Requirements (`requirements.txt`)
