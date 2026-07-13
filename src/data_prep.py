"""
data_prep.py
============
Transforms the raw lending portfolio XLSX into a JSONL instruction-tuning
dataset for fine-tuning Llama-3.2-3B-Instruct with QLoRA.

Pipeline stages:
  1. Load & inspect raw data
  2. Normalize dirty categorical fields (LOAN_PRODUCT, OCCUPATION, GENDER)
  3. Handle outliers and invalid values (CURRENT_DPD)
  4. Impute missing values with group-median strategy
  5. Derive engineered features (FOIR, Credit Utilization, IS_DELINQUENT, etc.)
  6. Derive task labels (RISK_LABEL, APPROVAL_LABEL)
  7. Generate instruction-tuning prompt-completion pairs for 3 task types
  8. Export train / val / test JSONL splits

Design decisions are documented inline. Every transformation choice that
could affect model behaviour is explained.
"""

import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "Lending_Loan_Portfolio_1000_Raw.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL = PROCESSED_DIR / "train.jsonl"
VAL_JSONL   = PROCESSED_DIR / "val.jsonl"
TEST_JSONL  = PROCESSED_DIR / "test.jsonl"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Loan_Portfolio_Raw", engine="openpyxl")
    print(f"[Load]  {len(df)} records, {df.shape[1]} columns")
    print(f"        Nulls per column:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: NORMALIZE DIRTY CATEGORICAL FIELDS
# ──────────────────────────────────────────────────────────────────────────────

# Decision: use mapping dicts rather than regex to make every transformation
# explicit and auditable. Any value not in the map is left as-is but flagged.

LOAN_PRODUCT_MAP = {
    # Home Loan variants
    "Home":          "Home Loan",
    "Housing Loan":  "Home Loan",
    "Mortgage":      "Home Loan",
    "HL":            "Home Loan",
    # Personal Loan variants
    "PL":            "Personal Loan",
    "Personal":      "Personal Loan",
    "P-Loan":        "Personal Loan",
    "Pers Loan":     "Personal Loan",
    # Vehicle Loan variants
    "Vehicle":       "Vehicle Loan",
    "VL":            "Vehicle Loan",
    "Auto Loan":     "Vehicle Loan",
    "Car Loan":      "Vehicle Loan",
    # MSME Loan variants
    "MSME":          "MSME Loan",
    "SME Loan":      "MSME Loan",
    "Enterprise Loan": "MSME Loan",
    # Consumer Durable Loan variants:
    "Consumer Loan":  "Consumer Durable Loan",
    "CD Loan":        "Consumer Durable Loan",
    "Durable Loan":   "Consumer Durable Loan",
}

OCCUPATION_MAP = {
    # Salaried variants:
    "SALARIED":           "Salaried",
    "salaried":           "Salaried",
    "Permanent Employee": "Salaried",
    "Employee":           "Salaried",
    # Business variants
    "BUSINESS":        "Business",
    "Shop Owner":      "Business",
    "Proprietor":      "Business",
    "Trader":          "Business",
    "Entrepreneur":    "Business",
    "Business Owner":  "Business",
    # Self Employed variants
    "SELF EMPLOYED":  "Self Employed",
    "Self-Employed":  "Self Employed",
    # Professional variants
    "PROFESSIONAL":  "Professional",
    "Lawyer":        "Professional",
    "Doctor":        "Professional",
}

GENDER_MAP = {
    "male":   "Male",
    "M":      "Male",
    "female": "Female",
    "F":      "Female",
}

CANONICAL_PRODUCTS = {
    "Personal Loan", "Vehicle Loan", "Home Loan",
    "MSME Loan", "Consumer Durable Loan"
}

CANONICAL_OCCUPATIONS = {"Salaried", "Business", "Self Employed", "Professional"}


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Loan product
    df["LOAN_PRODUCT"] = df["LOAN_PRODUCT"].map(
        lambda x: LOAN_PRODUCT_MAP.get(x, x) if pd.notna(x) else x
    )
    unknown_products = set(df["LOAN_PRODUCT"].dropna()) - CANONICAL_PRODUCTS
    if unknown_products:
        print(f"[Warn]  Unmapped LOAN_PRODUCT values: {unknown_products}")

    # Occupation — nulls become "Unknown"
    df["OCCUPATION"] = df["OCCUPATION"].map(
        lambda x: OCCUPATION_MAP.get(x, x) if pd.notna(x) else "Unknown"
    )
    unknown_occ = set(df["OCCUPATION"].dropna()) - CANONICAL_OCCUPATIONS - {"Unknown"}
    if unknown_occ:
        print(f"[Warn]  Unmapped OCCUPATION values: {unknown_occ}")

    # Gender
    df["GENDER"] = df["GENDER"].map(
        lambda x: GENDER_MAP.get(x, x) if pd.notna(x) else x
    )
    unknown_gender = set(df["GENDER"].dropna()) - {"Male", "Female"}
    if unknown_gender:
        print(f"[Warn]  Unmapped GENDER values: {unknown_gender}")

    print(f"[Norm]  LOAN_PRODUCT unique values: {sorted(df['LOAN_PRODUCT'].dropna().unique())}")
    print(f"[Norm]  OCCUPATION unique values:   {sorted(df['OCCUPATION'].unique())}")
    print(f"[Norm]  GENDER unique values:        {sorted(df['GENDER'].dropna().unique())}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: HANDLE OUTLIERS
# ──────────────────────────────────────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # CURRENT_DPD: negative values are operationally meaningless (possible
    # advance payment entries). Clip to [0, 365]. Values > 365 are likely
    # system error sentinels (e.g., 999 = "not recorded").
    n_negative = (df["CURRENT_DPD"] < 0).sum()
    n_extreme  = (df["CURRENT_DPD"] > 365).sum()
    df["CURRENT_DPD"] = df["CURRENT_DPD"].clip(lower=0, upper=365)
    print(f"[Outlier] CURRENT_DPD: clipped {n_negative} negative, {n_extreme} extreme (>365) values → [0, 365]")

    # MAX_DPD: similar treatment — cap at 365
    n_extreme_max = (df["MAX_DPD"] > 365).sum()
    df["MAX_DPD"] = df["MAX_DPD"].clip(lower=0, upper=365)
    if n_extreme_max:
        print(f"[Outlier] MAX_DPD: clipped {n_extreme_max} values > 365")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: IMPUTE MISSING VALUES
# ──────────────────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - MONTHLY_INCOME: group-median by (OCCUPATION, LOAN_PRODUCT). Rationale:
      income varies significantly by occupation type and the product they took.
    - BUREAU_SCORE: group-median by (LOAN_PRODUCT, COLLECTION_BUCKET). Rationale:
      delinquency bucket is a strong proxy for credit behavior when bureau score
      is missing.
    - LAST_PAYMENT_DATE: null means no payment recorded. Do not impute with a
      date — instead the derived IS_DELINQUENT flag will capture this correctly.
    """
    df = df.copy()

    # Track which rows were imputed for use in prompts
    df["INCOME_IMPUTED"]  = df["MONTHLY_INCOME"].isna().astype(int)
    df["BUREAU_IMPUTED"]  = df["BUREAU_SCORE"].isna().astype(int)

    # MONTHLY_INCOME — group median
    income_medians = df.groupby(["OCCUPATION", "LOAN_PRODUCT"])["MONTHLY_INCOME"].transform("median")
    global_income_median = df["MONTHLY_INCOME"].median()
    df["MONTHLY_INCOME"] = df["MONTHLY_INCOME"].fillna(income_medians).fillna(global_income_median)
    print(f"[Impute] MONTHLY_INCOME: filled {df['INCOME_IMPUTED'].sum()} nulls via group-median (OCCUPATION × LOAN_PRODUCT)")

    # BUREAU_SCORE — group median
    bureau_medians = df.groupby(["LOAN_PRODUCT", "COLLECTION_BUCKET"])["BUREAU_SCORE"].transform("median")
    global_bureau_median = df["BUREAU_SCORE"].median()
    df["BUREAU_SCORE"] = df["BUREAU_SCORE"].fillna(bureau_medians).fillna(global_bureau_median)
    print(f"[Impute] BUREAU_SCORE: filled {df['BUREAU_IMPUTED'].sum()} nulls via group-median (LOAN_PRODUCT × COLLECTION_BUCKET)")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # FOIR: Fixed Obligation to Income Ratio
    # Cap at 1.0 — values above 100% are unsustainable and likely data errors.
    df["FOIR"] = (df["EMI_AMOUNT"] / df["MONTHLY_INCOME"]).clip(upper=1.0).round(4)

    # EMI-to-Income Ratio: percentage form of FOIR for narrative generation
    df["EMI_INCOME_RATIO_PCT"] = (df["FOIR"] * 100).round(1)

    # Credit Utilization: what fraction of sanctioned amount remains outstanding
    # Guard against zero sanction amounts
    df["CREDIT_UTILIZATION"] = (
        df["OUTSTANDING_BALANCE"] / df["SANCTION_AMOUNT"].replace(0, np.nan)
    ).clip(upper=1.5).round(4)   # cap at 150% (can exceed 1.0 due to interest)

    # IS_DELINQUENT: binary flag for current active delinquency
    # Also mark records with null LAST_PAYMENT_DATE as delinquent
    df["IS_DELINQUENT"] = (
        (df["CURRENT_DPD"] > 0) | (df["LAST_PAYMENT_DATE"].isna())
    ).astype(int)

    # LOAN_TO_INCOME_RATIO: sanction amount relative to annual income
    df["LOAN_TO_INCOME"] = (df["SANCTION_AMOUNT"] / (df["MONTHLY_INCOME"] * 12)).round(2)

    print(f"[Feat]  Engineered features: FOIR, EMI_INCOME_RATIO_PCT, CREDIT_UTILIZATION, IS_DELINQUENT, LOAN_TO_INCOME")
    print(f"        FOIR > 0.60 (stress):  {(df['FOIR'] > 0.60).sum()} records ({(df['FOIR'] > 0.60).mean()*100:.1f}%)")
    print(f"        IS_DELINQUENT = 1:      {df['IS_DELINQUENT'].sum()} records")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: DERIVE TASK LABELS
# ──────────────────────────────────────────────────────────────────────────────

def bureau_tier(score: float) -> str:
    if score >= 800: return "Excellent (800+)"
    if score >= 750: return "Very Good (750–799)"
    if score >= 700: return "Good (700–749)"
    if score >= 650: return "Moderate Risk (650–699)"
    return "High Risk (<650)"


def derive_risk_label(row: pd.Series) -> str:
    """
    Composite rule-based risk classification.
    Priority order: hard negatives first, then graduated signals.

    High Risk triggers (any one sufficient):
      - Known default or write-off (ground truth)
      - Bureau score < 650
      - Current DPD > 60 (serious delinquency)
      - Null LAST_PAYMENT_DATE (no payment on record)
      - Collection bucket 90+ (near-default, per system-prompt definition)

    Medium Risk triggers (any one, no High triggers):
      - Bureau score 650–699
      - Current DPD 1–60
      - FOIR > 0.60 (financial stress)
      - Max DPD > 90 in history (prior near-default)
      - Collection bucket 31-60 or 61-90

    Low Risk: none of the above.
    """
    bureau = row["BUREAU_SCORE"]
    dpd    = row["CURRENT_DPD"]
    foir   = row["FOIR"]
    max_dpd = row["MAX_DPD"]
    bucket  = row["COLLECTION_BUCKET"]

    # Hard High Risk
    if row["DEFAULT_FLAG"] == 1 or row["WRITE_OFF_FLAG"] == 1:
        return "High Risk"
    if bureau < 650:
        return "High Risk"
    if dpd > 60:
        return "High Risk"
    if pd.isna(row["LAST_PAYMENT_DATE"]) and row["LOAN_STATUS"] == "Active":
        return "High Risk"
    # 90+ collection bucket means the account has, at some point, crossed the
    # near-default threshold described in the system prompt — treat as High
    # Risk even if CURRENT_DPD has since been reset (e.g. a partial catch-up
    # payment), consistent with dpd > 60 above.
    if bucket == "90+":
        return "High Risk"

    # Medium Risk
    if 650 <= bureau < 700:
        return "Medium Risk"
    if 0 < dpd <= 60:
        return "Medium Risk"
    if foir > 0.60:
        return "Medium Risk"
    if max_dpd > 90:
        return "Medium Risk"
    if bucket in ("31-60", "61-90"):
        return "Medium Risk"

    return "Low Risk"


def derive_approval_label(row: pd.Series) -> str:
    """
    Approval recommendation for a new loan application from this borrower.

    Reject: prior default/write-off, or bureau < 650, or FOIR > 0.65,
            or serious active delinquency (DPD > 60)
    Approve: low risk, strong bureau (>=750), FOIR < 0.40, DPD = 0
    Approve with Conditions: everything in between
    """
    bureau = row["BUREAU_SCORE"]
    foir   = row["FOIR"]
    dpd    = row["CURRENT_DPD"]

    # Hard rejections
    if row["DEFAULT_FLAG"] == 1 or row["WRITE_OFF_FLAG"] == 1:
        return "Reject"
    if bureau < 650:
        return "Reject"
    if foir > 0.65:
        return "Reject"
    if dpd > 60:
        return "Reject"

    # Clean approvals
    if bureau >= 750 and foir < 0.40 and dpd == 0 and row["MAX_DPD"] <= 30:
        return "Approve"

    # Everything else: conditional
    return "Approve with Conditions"


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # DEFAULT_FLAG/WRITE_OFF_FLAG nulls would silently evaluate as "no default"
    # in the `== 1` checks below — treat missing as not-defaulted explicitly.
    n_null_flags = df["DEFAULT_FLAG"].isna().sum() + df["WRITE_OFF_FLAG"].isna().sum()
    if n_null_flags:
        print(f"[Warn]  {n_null_flags} null DEFAULT_FLAG/WRITE_OFF_FLAG values — filling with 0")
    df["DEFAULT_FLAG"]   = df["DEFAULT_FLAG"].fillna(0).astype(int)
    df["WRITE_OFF_FLAG"] = df["WRITE_OFF_FLAG"].fillna(0).astype(int)

    df["RISK_LABEL"]     = df.apply(derive_risk_label, axis=1)
    df["APPROVAL_LABEL"] = df.apply(derive_approval_label, axis=1)

    print(f"\n[Labels] RISK distribution:")
    print(df["RISK_LABEL"].value_counts().to_string())
    print(f"\n[Labels] APPROVAL distribution:")
    print(df["APPROVAL_LABEL"].value_counts().to_string())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: PROMPT-COMPLETION GENERATION
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Lending Intelligence Assistant at ABC Finance Ltd., a leading retail lending organisation. You are an expert in credit underwriting, portfolio risk management, and lending operations.

You understand and apply the following domain-specific concepts:

BUREAU SCORE TIERS:
- 800+:     Excellent credit — very low default risk
- 750–799:  Very Good — low risk, eligible for best rates
- 700–749:  Good — acceptable risk, standard terms
- 650–699:  Moderate Risk — careful scrutiny required
- Below 650: High Risk — elevated probability of default

DPD (Days Past Due) — days a borrower is overdue on an EMI:
- 0:       On-time payment
- 1–30:    Slight delay — early warning signal
- 31–60:   Moderate delinquency — active monitoring required
- 61–90:   Serious delinquency — collections intervention likely
- >90:     Potential default — escalation required

FOIR (Fixed Obligation to Income Ratio) = EMI ÷ Monthly Income:
- <0.40:   Comfortable — strong repayment capacity
- 0.40–0.60: Moderate — acceptable but watch for stress
- >0.60:   Financial Stress — high repayment burden

COLLECTION BUCKETS (based on DPD):
- Current: 0 DPD — healthy
- 1-30:    1–30 days overdue
- 31-60:   31–60 days overdue
- 61-90:   61–90 days overdue
- 90+:     Over 90 days — near default

CREDIT UTILIZATION = Outstanding Balance ÷ Sanctioned Amount:
- <50%:  Low utilization
- 50–80%: Moderate utilization
- >80%:  High utilization — financial stress signal

Always provide precise, actionable assessments grounded in these definitions. Never hallucinate values — only use figures from the borrower profile provided."""


def fmt_inr(value: float) -> str:
    """Format a number as Indian Rupees with comma separators."""
    try:
        return f"₹{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def foir_narrative(foir: float) -> str:
    if foir < 0.40:
        return f"FOIR of {foir:.2f} ({fmt_pct(foir*100)}) is comfortable — well within the safe threshold"
    if foir < 0.60:
        return f"FOIR of {foir:.2f} ({fmt_pct(foir*100)}) is moderate — approaching but within acceptable limits"
    return f"FOIR of {foir:.2f} ({fmt_pct(foir*100)}) exceeds the 0.60 financial stress threshold"


def bureau_narrative(score: int) -> str:
    tier = bureau_tier(score)
    if score >= 800:
        return f"Bureau score of {score} ({tier}) reflects excellent credit discipline"
    if score >= 750:
        return f"Bureau score of {score} ({tier}) reflects strong creditworthiness"
    if score >= 700:
        return f"Bureau score of {score} ({tier}) is acceptable for standard lending"
    if score >= 650:
        return f"Bureau score of {score} ({tier}) warrants careful scrutiny"
    return f"Bureau score of {score} ({tier}) places this borrower in the high-risk tier"


def dpd_narrative(dpd: int, max_dpd: int) -> str:
    if dpd == 0 and max_dpd == 0:
        return "Repayment behaviour is exemplary — no delinquency on record (Current DPD: 0, Max DPD: 0)"
    if dpd == 0:
        return f"Currently on-time (DPD: 0), though lifetime Max DPD of {max_dpd} days signals prior payment stress"
    return f"Currently {dpd} days overdue (DPD: {dpd}), with a lifetime Max DPD of {max_dpd} days"


def utilization_narrative(util: float) -> str:
    pct = util * 100
    if pct < 50:
        return f"Credit utilization is low at {pct:.1f}%"
    if pct < 80:
        return f"Credit utilization is moderate at {pct:.1f}%"
    return f"Credit utilization is elevated at {pct:.1f}% — indicating reduced financial buffer"


# ── Prompt template sets (3 per task for diversity) ──

SUMMARY_PROMPTS = [
    "Generate a structured loan summary for the following borrower profile. Include repayment behaviour, credit health, and key risk signals.",
    "Summarise this borrower's loan profile for an underwriter review. Highlight the most significant risk indicators and overall account health.",
    "Provide a credit narrative for the following customer account. The summary should be suitable for a credit officer making a portfolio decision.",
]

RISK_PROMPTS = [
    "Classify the credit risk for the following borrower. Respond with Low Risk, Medium Risk, or High Risk, followed by a detailed justification citing specific metrics.",
    "What is the risk category of this customer? Assess their bureau score, DPD history, FOIR, and delinquency behaviour, then state your classification.",
    "Perform a credit risk assessment for this borrower. State the risk classification (Low Risk / Medium Risk / High Risk) and explain the key drivers.",
]

APPROVAL_PROMPTS = [
    "Provide a loan approval recommendation for this application. Choose from: Approve, Approve with Conditions, or Reject. Justify your decision with specific metrics.",
    "Should this loan be approved? Analyse the borrower profile and make a recommendation. If conditions are required, state them explicitly.",
    "As a credit underwriter, evaluate this loan application and recommend an appropriate decision: Approve, Approve with Conditions, or Reject.",
]


def build_borrower_context(row: pd.Series) -> str:
    """Build the structured input block included in every prompt."""
    lines = [
        f"Borrower Profile:",
        f"  Age: {int(row['AGE'])} | Gender: {row['GENDER']} | State: {row['STATE']}",
        f"  Occupation: {row['OCCUPATION']} | Monthly Income: {fmt_inr(row['MONTHLY_INCOME'])}",
        f"",
        f"Loan Details:",
        f"  Product: {row['LOAN_PRODUCT']} | Loan ID: {row['LOAN_ID']}",
        f"  Requested: {fmt_inr(row['LOAN_AMOUNT'])} | Sanctioned: {fmt_inr(row['SANCTION_AMOUNT'])}",
        f"  Tenure: {int(row['LOAN_TENURE'])} months | Interest Rate: {row['INTEREST_RATE']}% p.a.",
        f"  EMI: {fmt_inr(row['EMI_AMOUNT'])} | Outstanding Balance: {fmt_inr(row['OUTSTANDING_BALANCE'])}",
        f"",
        f"Credit & Repayment:",
        f"  Bureau Score: {int(row['BUREAU_SCORE'])} ({bureau_tier(row['BUREAU_SCORE'])})",
        f"  Current DPD: {int(row['CURRENT_DPD'])} days | Max DPD (lifetime): {int(row['MAX_DPD'])} days",
        f"  Collection Bucket: {row['COLLECTION_BUCKET']} | Loan Status: {row['LOAN_STATUS']}",
        f"  Default Flag: {'Yes' if row['DEFAULT_FLAG'] == 1 else 'No'} | Write-Off Flag: {'Yes' if row['WRITE_OFF_FLAG'] == 1 else 'No'}",
        f"",
        f"Derived Metrics:",
        f"  FOIR: {row['FOIR']:.2f} ({fmt_pct(row['EMI_INCOME_RATIO_PCT'])} of monthly income committed to EMI)",
        f"  Credit Utilization: {fmt_pct(row['CREDIT_UTILIZATION']*100)} (outstanding vs sanctioned)",
        f"  Loan-to-Annual-Income: {row['LOAN_TO_INCOME']:.2f}x",
        f"  Currently Delinquent: {'Yes' if row['IS_DELINQUENT'] == 1 else 'No'}",
    ]
    if row.get("INCOME_IMPUTED", 0) == 1:
        lines.append("  [Note: Monthly income was not reported — estimated from peer group median]")
    if row.get("BUREAU_IMPUTED", 0) == 1:
        lines.append("  [Note: Bureau score was not available — estimated from segment median]")
    return "\n".join(lines)


# ── Completion generators ──

def generate_summary_completion(row: pd.Series, template_idx: int) -> str:
    risk     = row["RISK_LABEL"]
    approval = row["APPROVAL_LABEL"]
    bureau   = int(row["BUREAU_SCORE"])
    foir     = row["FOIR"]
    dpd      = int(row["CURRENT_DPD"])
    max_dpd  = int(row["MAX_DPD"])
    util     = row["CREDIT_UTILIZATION"]
    product  = row["LOAN_PRODUCT"]
    occ      = row["OCCUPATION"]
    outstanding = row["OUTSTANDING_BALANCE"]
    sanction    = row["SANCTION_AMOUNT"]

    overall_health = {
        "Low Risk": "HEALTHY — low risk portfolio account",
        "Medium Risk": "MONITOR — moderate risk, requires periodic review",
        "High Risk": "CRITICAL — high risk account, immediate attention required",
    }[risk]

    completion = (
        f"{occ} borrower carrying a {product}. "
        f"{bureau_narrative(bureau)}. "
        f"{dpd_narrative(dpd, max_dpd)}. "
        f"{foir_narrative(foir)}.\n\n"
    )

    completion += f"Outstanding balance of {fmt_inr(outstanding)} against a sanctioned limit of {fmt_inr(sanction)}. "
    completion += f"{utilization_narrative(util)}.\n\n"

    if row["DEFAULT_FLAG"] == 1:
        completion += "⚠️  Account has defaulted. Recovery proceedings should be assessed. "
    if row["WRITE_OFF_FLAG"] == 1:
        completion += "⚠️  Loan has been written off. Probability of full recovery is low. "

    completion += f"\nRisk Classification: {risk}\n"
    completion += f"Recommended Action: {approval}\n"
    completion += f"Overall Account Status: {overall_health}"

    return completion.strip()


def generate_risk_completion(row: pd.Series, template_idx: int) -> str:
    risk    = row["RISK_LABEL"]
    bureau  = int(row["BUREAU_SCORE"])
    foir    = row["FOIR"]
    dpd     = int(row["CURRENT_DPD"])
    max_dpd = int(row["MAX_DPD"])
    bucket  = row["COLLECTION_BUCKET"]

    reasons = []

    # Bureau score analysis
    if bureau < 650:
        reasons.append(f"Bureau score of {bureau} falls below the 650 high-risk threshold, indicating significant historical repayment issues")
    elif bureau < 700:
        reasons.append(f"Bureau score of {bureau} (Moderate Risk tier, 650–699) warrants careful monitoring")
    elif bureau < 750:
        reasons.append(f"Bureau score of {bureau} (Good tier, 700–749) is acceptable but not strong")
    else:
        reasons.append(f"Bureau score of {bureau} ({bureau_tier(bureau)}) demonstrates strong credit discipline")

    # DPD analysis
    if dpd > 60:
        reasons.append(f"Current DPD of {dpd} days indicates serious active delinquency — collections intervention is required")
    elif dpd > 30:
        reasons.append(f"Current DPD of {dpd} days represents moderate delinquency — escalating repayment stress")
    elif dpd > 0:
        reasons.append(f"Current DPD of {dpd} days is a mild delay but signals emerging payment difficulty")
    else:
        reasons.append(f"Current DPD of 0 — the most recent payment was received on time")

    # Historical DPD
    if max_dpd > 90:
        reasons.append(f"Lifetime Max DPD of {max_dpd} days indicates a prior near-default episode")
    elif max_dpd > 60:
        reasons.append(f"Lifetime Max DPD of {max_dpd} days shows prior serious delinquency in account history")

    # FOIR
    if foir > 0.60:
        reasons.append(f"FOIR of {foir:.2f} ({foir*100:.1f}%) exceeds the 0.60 financial stress threshold — repayment capacity is strained")
    elif foir > 0.40:
        reasons.append(f"FOIR of {foir:.2f} ({foir*100:.1f}%) is moderate — repayment capacity exists but is not comfortable")
    else:
        reasons.append(f"FOIR of {foir:.2f} ({foir*100:.1f}%) is well within the safe zone — strong repayment capacity")

    # Default / write-off
    if row["DEFAULT_FLAG"] == 1:
        reasons.append("Default flag is active — borrower has previously failed to meet payment obligations")
    if row["WRITE_OFF_FLAG"] == 1:
        reasons.append("Write-off flag is active — lender has treated this loan as a loss")

    # Collection bucket context
    bucket_context = {
        "Current":  "Account is current — all payments up to date",
        "1-30":     "Account is in the 1-30 DPD bucket — early stage delinquency requiring monitoring",
        "31-60":    "Account is in the 31-60 DPD bucket — active delinquency, collections outreach warranted",
        "61-90":    "Account is in the 61-90 DPD bucket — serious delinquency, escalation recommended",
        "90+":      "Account has crossed the 90+ DPD threshold — near-default status",
    }
    if bucket in bucket_context:
        reasons.append(bucket_context[bucket])

    # Build completion
    completion = f"Risk Classification: {risk.upper()}\n\n"
    completion += "Analysis:\n"
    for i, reason in enumerate(reasons, 1):
        completion += f"  {i}. {reason}.\n"

    action_map = {
        "Low Risk":    "No immediate action required. Standard portfolio monitoring applies.",
        "Medium Risk": "Flag for quarterly review. Monitor DPD trend and FOIR for deterioration.",
        "High Risk":   "Escalate immediately. Initiate collections contact and assess recovery options.",
    }
    completion += f"\nRecommended Action: {action_map[risk]}"

    return completion.strip()


def generate_approval_completion(row: pd.Series, template_idx: int) -> str:
    approval = row["APPROVAL_LABEL"]
    risk     = row["RISK_LABEL"]
    bureau   = int(row["BUREAU_SCORE"])
    foir     = row["FOIR"]
    dpd      = int(row["CURRENT_DPD"])
    max_dpd  = int(row["MAX_DPD"])
    util     = row["CREDIT_UTILIZATION"]
    product  = row["LOAN_PRODUCT"]

    completion = f"Recommendation: {approval.upper()}\n\n"

    strengths = []
    concerns  = []

    # Strengths
    if bureau >= 750:
        strengths.append(f"Bureau score of {bureau} ({bureau_tier(bureau)}) demonstrates strong creditworthiness")
    elif bureau >= 700:
        strengths.append(f"Bureau score of {bureau} ({bureau_tier(bureau)}) meets the standard threshold")
    if foir < 0.40:
        strengths.append(f"FOIR of {foir:.2f} is well within the comfortable zone — strong repayment capacity")
    elif foir < 0.50:
        strengths.append(f"FOIR of {foir:.2f} is moderate — repayment capacity is adequate")
    if dpd == 0:
        strengths.append("No current delinquency — latest EMI payment received on time")
    if max_dpd == 0:
        strengths.append("Zero lifetime delinquency — exemplary repayment discipline across the loan lifecycle")
    if row["DEFAULT_FLAG"] == 0 and row["WRITE_OFF_FLAG"] == 0:
        strengths.append("No default or write-off history")

    # Concerns
    if bureau < 700:
        concerns.append(f"Bureau score of {bureau} falls below the 700 preferred threshold for {product}")
    if foir >= 0.60:
        concerns.append(f"FOIR of {foir:.2f} exceeds the 0.60 financial stress threshold — repayment capacity is constrained")
    elif foir >= 0.45:
        concerns.append(f"FOIR of {foir:.2f} is approaching the moderate-stress zone")
    if dpd > 0:
        concerns.append(f"Current DPD of {dpd} days — active delinquency on existing obligations")
    if max_dpd > 60:
        concerns.append(f"Lifetime Max DPD of {max_dpd} days signals prior serious delinquency")
    if util > 0.80:
        concerns.append(f"Credit utilization of {util*100:.1f}% is elevated — limited financial headroom")
    if row["DEFAULT_FLAG"] == 1:
        concerns.append("Prior default on record — significant negative indicator")
    if row["WRITE_OFF_FLAG"] == 1:
        concerns.append("Prior write-off on record — severe credit risk signal")

    if strengths:
        completion += "Strengths:\n"
        for s in strengths:
            completion += f"  + {s}\n"
        completion += "\n"

    if concerns:
        completion += "Concerns:\n"
        for c in concerns:
            completion += f"  - {c}\n"
        completion += "\n"

    # Conditions for conditional approvals
    if approval == "Approve with Conditions":
        conditions = []
        if foir >= 0.45:
            capped_emi = row["MONTHLY_INCOME"] * 0.40
            capped_amount = capped_emi * row["LOAN_TENURE"] * 0.85  # rough principal estimate
            conditions.append(f"Cap sanctioned amount at approximately {fmt_inr(capped_amount)} to bring FOIR below 0.40")
        if bureau < 720:
            conditions.append("Require last 6 months' bank statements and income verification")
        if max_dpd > 30:
            conditions.append("Mandate post-disbursement monitoring for first 12 EMI cycles")
        if util > 0.70:
            conditions.append("Review existing loan consolidation options to reduce utilization before disbursement")
        if not conditions:
            conditions.append("Standard enhanced due diligence and quarterly portfolio review for the first year")
        completion += "Conditions:\n"
        for i, cond in enumerate(conditions, 1):
            completion += f"  {i}. {cond}\n"
        completion += "\n"

    if approval == "Reject":
        completion += "Rejection rationale: "
        if row["DEFAULT_FLAG"] == 1 or row["WRITE_OFF_FLAG"] == 1:
            completion += "Prior default or write-off creates unacceptable credit risk. Eligibility can be reconsidered after 24 months of clean repayment history."
        elif bureau < 650:
            completion += f"Bureau score of {bureau} falls below the minimum threshold of 650 for {product} eligibility."
        elif foir > 0.65:
            completion += f"FOIR of {foir:.2f} exceeds the maximum permissible limit of 0.65, indicating unsustainable debt burden."
        else:
            completion += "Multiple risk signals collectively indicate unacceptable credit risk at this time."

    return completion.strip()


# ── Dataset assembly ──

def build_examples(df: pd.DataFrame) -> list[dict]:
    """
    Generate 3 instruction-tuning examples per record — one per task type.
    Prompt template is cycled per-record for diversity.
    Returns list of dicts in HuggingFace messages format.
    """
    examples = []

    for enum_idx, (_, row) in enumerate(df.iterrows()):
        context = build_borrower_context(row)
        template_idx = enum_idx % 3  # cycle through 3 prompt variants

        tasks = [
            (SUMMARY_PROMPTS[template_idx],  generate_summary_completion),
            (RISK_PROMPTS[template_idx],     generate_risk_completion),
            (APPROVAL_PROMPTS[template_idx], generate_approval_completion),
        ]

        for instruction, completion_fn in tasks:
            user_content   = f"{instruction}\n\n{context}"
            asst_content   = completion_fn(row, template_idx)

            example = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": asst_content},
                ]
            }
            examples.append(example)

    print(f"[Prompts] Generated {len(examples)} prompt-completion pairs ({len(df)} records × 3 tasks)")
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8: SPLIT & EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def split_and_export(df: pd.DataFrame, examples: list[dict]) -> None:
    """
    Stratified 80/10/10 split on RISK_LABEL.
    We split at the record level first, then map to examples.
    This ensures the same record doesn't appear in both train and eval.
    """
    # Create index mapping: each record produces 3 consecutive examples
    record_indices = list(range(len(df)))

    # Stratify by RISK_LABEL
    train_idx, temp_idx = train_test_split(
        record_indices,
        test_size=0.20,
        stratify=df["RISK_LABEL"].tolist(),
        random_state=RANDOM_SEED,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=df.iloc[temp_idx]["RISK_LABEL"].tolist(),
        random_state=RANDOM_SEED,
    )

    def records_to_examples(indices: list[int]) -> list[dict]:
        out = []
        for i in indices:
            out.extend(examples[i * 3: i * 3 + 3])
        return out

    train_examples = records_to_examples(train_idx)
    val_examples   = records_to_examples(val_idx)
    test_examples  = records_to_examples(test_idx)

    random.shuffle(train_examples)

    def write_jsonl(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(train_examples, TRAIN_JSONL)
    write_jsonl(val_examples,   VAL_JSONL)
    write_jsonl(test_examples,  TEST_JSONL)

    print(f"\n[Export] train.jsonl → {len(train_examples)} examples ({len(train_idx)} records)")
    print(f"[Export] val.jsonl   → {len(val_examples)} examples ({len(val_idx)} records)")
    print(f"[Export] test.jsonl  → {len(test_examples)} examples ({len(test_idx)} records)")
    print(f"[Export] All files saved to {PROCESSED_DIR}")

    # Also export cleaned CSV for reference
    df.to_csv(PROCESSED_DIR / "lending_cleaned.csv", index=False)
    print(f"[Export] lending_cleaned.csv saved ({len(df)} records, {df.shape[1]} columns)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    print("=" * 60)
    print("  LENDING AI — DATA PREPARATION PIPELINE")
    print("=" * 60)

    df = load_raw_data(RAW_PATH)

    print("\n── Step 2: Normalise categoricals ──")
    df = normalize_categoricals(df)

    print("\n── Step 3: Handle outliers ──")
    df = handle_outliers(df)

    print("\n── Step 4: Impute missing values ──")
    df = impute_missing(df)

    print("\n── Step 5: Feature engineering ──")
    df = engineer_features(df)

    print("\n── Step 6: Derive task labels ──")
    df = derive_labels(df)

    print("\n── Step 7: Generate prompt-completion pairs ──")
    examples = build_examples(df)

    print("\n── Step 8: Split and export ──")
    split_and_export(df, examples)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_pipeline()
