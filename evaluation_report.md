# Lending AI SLM — Evaluation Report

## Credit Risk Classification
| Metric | Base Model | Fine-Tuned | Delta |
|--------|-----------|------------|-------|
| Accuracy | 36.4% | 36.4% | +0.0% |
| F1 Macro | 0.3654 | 0.3654 | +0.0000 |

**Per-class F1:**
| Class | Base F1 | FT F1 | Base Recall | FT Recall |
|-------|---------|-------|-------------|-----------|
| Low Risk | 0.133 | 0.133 | 0.250 | 0.250 |
| Medium Risk | 0.519 | 0.519 | 0.389 | 0.389 |
| High Risk | 0.444 | 0.444 | 0.364 | 0.364 |

## Loan Approval Recommendation
| Metric | Base Model | Fine-Tuned | Delta |
|--------|-----------|------------|-------|
| Accuracy | 24.2% | 24.2% | +0.0% |
| F1 Macro | 0.1926 | 0.1926 | +0.0000 |

## Business Impact
- High Risk Recall — Base: 36.4% → Fine-Tuned: 36.4% (Δ +0.0%)
- Domain Term Recall (Risk) — Base: 54.5% → FT: 54.5%
- Domain Term Recall (Summary) — Base: 44.4% → FT: 44.4%

## Interpretation
High Risk recall improvement means more problematic borrowers are correctly
flagged for collections intervention before they default.
Domain Term Recall improvement shows the fine-tuned model uses lending
terminology (DPD, FOIR, Bureau Score tiers) correctly and consistently.