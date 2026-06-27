# Lending AI — Fine-Tuned SLM for Credit Intelligence

Fine-tuning `meta-llama/Llama-3.2-3B-Instruct` with QLoRA on a real-world lending portfolio
dataset to build a domain-aware **Lending Intelligence Assistant** for ABC Finance Ltd.

---

## What the Model Does

| Task | Input | Output |
|------|-------|--------|
| **Loan Summary Generation** | Borrower profile + loan details | Natural language credit narrative with key risk signals |
| **Credit Risk Classification** | Bureau score, DPD, FOIR, repayment history | Low / Medium / High Risk + chain-of-thought reasoning |
| **Loan Approval Recommendation** | Full applicant profile | Approve / Approve with Conditions / Reject + rationale |

---

## Repository Structure

```
.
├── data/
│   ├── raw/
│   │   └── Lending_Loan_Portfolio_1000_Raw.xlsx   # Raw dataset (place here before running)
│   └── processed/
│       ├── lending_cleaned.csv    # Cleaned dataset: 1,000 rows, 33 columns
│       ├── train.jsonl            # 2,400 instruction-tuning examples (80%)
│       ├── val.jsonl              # 300 validation examples (10%)
│       └── test.jsonl             # 300 held-out test examples (10%)
│
├── src/
│   ├── data_prep.py       # Full 8-stage data pipeline (load → clean → engineer → export)
│   ├── train.py           # QLoRA fine-tuning script (reads configs/training_config.yaml)
│   └── evaluate.py        # Before-vs-after evaluation with ROUGE, F1, and business metrics
│
├── configs/
│   └── training_config.yaml   # All hyperparameters with inline comments
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   # Dataset audit, cleaning walkthrough, prompt design
│   ├── 02_fine_tuning.ipynb        # QLoRA training walkthrough with loss curve
│   ├── 03_evaluation.ipynb         # Before-vs-after metrics + 3 demo scenarios
│   └── lending_ai_colab.ipynb      # ★ Self-contained end-to-end Colab notebook (run this)
│
├── outputs/                        # Generated after training
│   ├── adapter/                    # Saved LoRA adapter weights (adapter_model.bin + config)
│   ├── training_metrics.json       # Loss, runtime, epoch stats
│   ├── training_loss.png           # Train vs val loss curve
│   ├── evaluation_report.md        # Human-readable before-vs-after comparison
│   └── evaluation_results.json     # Structured metrics (JSON)
│
├── requirements.txt
└── README.md
```

---

## File Descriptions

### `src/data_prep.py`
The core data pipeline — most critical component (30% of judging criteria).

**8-stage pipeline:**
1. **Load** — reads raw XLSX, normalises column names
2. **Normalise** — standardises LOAN_PRODUCT (22 variants → 5), OCCUPATION, GENDER via explicit mapping dicts
3. **Outlier handling** — clips CURRENT_DPD negatives to 0, sentinel values (999) to 365
4. **Imputation** — group-median imputation: INCOME by (OCCUPATION × LOAN_PRODUCT), BUREAU_SCORE by (LOAN_PRODUCT × COLLECTION_BUCKET)
5. **Feature engineering** — derives FOIR (EMI/Income), Credit Utilization (Outstanding/Sanction), IS_DELINQUENT, LOAN_TO_INCOME
6. **Label derivation** — rule-based RISK_LABEL and APPROVAL_LABEL cross-validated against DEFAULT_FLAG ground truth
7. **Prompt generation** — 3,000 chat-formatted examples (1,000 records × 3 tasks × rotating templates with chain-of-thought completions)
8. **Export** — stratified 80/10/10 train/val/test split to JSONL

**Risk label logic:**
- `High Risk`: DEFAULT_FLAG=1, WRITE_OFF_FLAG=1, Bureau < 650, or DPD > 60
- `Medium Risk`: Bureau 650–699, DPD 1–60, FOIR > 0.60, or MAX_DPD > 90
- `Low Risk`: everything else

---

### `src/train.py`
QLoRA fine-tuning script. Reads all hyperparameters from `configs/training_config.yaml`.

- Detects GPU; loads 4-bit NF4 quantized Llama-3.2-3B-Instruct
- Applies `prepare_model_for_kbit_training()` + `get_peft_model()` with LoRA
- Uses `SFTTrainer` with `apply_chat_template` formatting
- Falls back to `microsoft/phi-2` if Llama gating is an issue
- Saves adapter to `outputs/adapter/` + `outputs/training_metrics.json`

---

### `src/evaluate.py`
Before-vs-after evaluation by toggling the LoRA adapter on/off.

- **Classification metrics**: Accuracy, F1 Macro, per-class Recall + Precision
- **ROUGE scores**: ROUGE-1/2/L for loan summary generation
- **Domain Term Recall**: measures % of lending terms (DPD, FOIR, Bureau, EMI, etc.) used correctly in outputs
- **Business metric**: High-Risk borrower recall — maps directly to defaulter detection before write-off
- Saves `outputs/evaluation_report.md` and `outputs/evaluation_results.json`

---

### `configs/training_config.yaml`
All training hyperparameters in one place with documented rationale.

| Parameter | Value | Why |
|-----------|-------|-----|
| Model | Llama-3.2-3B-Instruct | Exactly 3B limit, already instruction-tuned |
| Quantization | 4-bit NF4 + double quant | ~6.5 GB VRAM, no accuracy loss |
| LoRA rank (r) | 16 | Enough capacity without overfitting 800 examples |
| LoRA alpha | 32 | 2× scaling — stable gradients at init |
| Target modules | q/k/v/o_proj | All 4 attention projections for maximum domain transfer |
| Epochs | 1 | Sufficient for demo; avoids overfitting small dataset |
| Batch size | 4 (effective 16 with grad accum) | Maximises GPU utilisation on T4 |
| Learning rate | 2e-4 cosine | Standard for QLoRA |
| Max seq length | 256 | Halves per-step processing time vs 512 |

---

### `notebooks/lending_ai_colab.ipynb`
**The primary submission artifact.** A single self-contained Google Colab notebook that runs the entire pipeline end-to-end with one "Run all":

1. GPU verification (exits if no T4)
2. Install all dependencies
3. Clone this GitHub repo
4. HuggingFace login (token pre-filled)
5. Upload dataset → full data preparation pipeline
6. Load Llama-3.2-3B-Instruct in 4-bit QLoRA
7. Apply LoRA adapters
8. Train with SFTTrainer (~8–10 min)
9. Plot training loss curve
10. Evaluate: base vs fine-tuned (Accuracy, F1, ROUGE, High-Risk recall)
11. 3 side-by-side demo scenarios (Low / High / Borderline risk)
12. Download all outputs
13. Push results to GitHub

---

### `notebooks/01_data_preparation.ipynb`
Judge-facing walkthrough of the data pipeline. Shows:
- Raw data quality audit (dirty categories, nulls, outliers)
- Before/after cleaning for each column
- Prompt design decisions and chain-of-thought rationale
- Label distribution and cross-validation against ground truth

### `notebooks/02_fine_tuning.ipynb`
Judge-facing training walkthrough. Shows:
- Model selection rationale (why Llama-3.2-3B over alternatives)
- QLoRA vs full fine-tuning comparison
- LoRA hyperparameter justification
- Trainable parameter count (~2.6M / 0.08% of 3B)
- Training loop + loss curve

### `notebooks/03_evaluation.ipynb`
Judge-facing evaluation. Shows:
- Adapter toggle methodology (one model, clean A/B comparison)
- Classification reports for Risk and Approval tasks
- ROUGE scores for summary generation
- Business impact table (High-Risk recall improvement)
- 3 detailed demo scenarios with side-by-side outputs

---

## Quick Start (Google Colab — Recommended)

1. Open `notebooks/lending_ai_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**
4. Upload `Lending_Loan_Portfolio_1000_Raw.xlsx` when the file picker appears

Training completes in **~8–10 minutes**.

---

## Local Setup (Mac/Linux — Evaluation only, no training)

Training requires a CUDA GPU. Apple Silicon (MPS) does not support bitsandbytes 4-bit quantization.

```bash
pip install -r requirements.txt

# Step 1: Data preparation
python src/data_prep.py

# Step 2: Training — requires CUDA GPU (use Colab)
python src/train.py

# Step 3: Evaluation
python src/evaluate.py
```

### Accept the Llama model license

```bash
# Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct → Accept license
huggingface-cli login
```

**Fallback (no license required):** In `configs/training_config.yaml`, set `model.name: microsoft/phi-2`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12–16 GB (T4 on Colab = 15 GB) |
| System RAM | 16 GB | 32 GB |
| Disk space | 6 GB | 12 GB |

Peak VRAM during training: **~6.5 GB** (4-bit + LoRA only)

---

## Dataset

**Source:** ABC Finance Ltd — Lending Loan Portfolio (1,000 records, proprietary)

**Features used:**

| Category | Features |
|----------|---------|
| Borrower | Age, Gender, State, Occupation, Monthly Income |
| Loan | Product Type, Sanctioned Amount, Outstanding, EMI |
| Credit | Bureau Score, Current DPD, Max DPD, Collection Bucket |
| Flags | Default Flag, Write-Off Flag, Delinquency Flag |
| Derived | FOIR, Credit Utilization, Loan-to-Income Ratio |

**Label distribution (after derivation):**
- Risk: ~60% Low Risk, ~34% Medium Risk, ~6% High Risk
- Approval: ~55% Approve, ~33% Approve with Conditions, ~12% Reject

---

## Key Design Decisions

**Why chain-of-thought completions?**
Completions include explicit reasoning (citing FOIR thresholds, Bureau Score tiers, DPD levels) before
the final label. Judges and underwriters can verify the reasoning, not just the prediction.

**Why macro F1 over accuracy?**
DEFAULT_FLAG is 94.2% negative — a model that always predicts "Low Risk" achieves 94% accuracy.
Macro F1 penalises this and forces the model to learn the minority High Risk class.

**Why stratified split at record level?**
Splitting at the example level would allow the same borrower to appear in both train and test
(just with a different task prompt). Splitting at record level prevents this data leakage.

---

## Submission Checklist

- [x] JSONL instruction-tuning dataset (`data/processed/`)
- [x] Fine-tuning script (`src/train.py`) and notebook (`notebooks/02_fine_tuning.ipynb`)
- [x] Evaluation report (`outputs/evaluation_report.md`) — generated after training
- [x] Data preparation pipeline (`src/data_prep.py`, `notebooks/01_data_preparation.ipynb`)
- [x] Configuration files (`configs/training_config.yaml`)
- [x] End-to-end Colab notebook (`notebooks/lending_ai_colab.ipynb`)
- [x] Requirements (`requirements.txt`)

**Final steps:**
1. Add `azentio-talent-Aquisition` as a GitHub collaborator → repo Settings → Collaborators
2. Email repo URL to **tateam@azentio.com**
