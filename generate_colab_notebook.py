"""
Generates the complete self-contained Colab notebook as lending_ai_colab.ipynb
Run once: python generate_colab_notebook.py
"""
import json

HF_TOKEN   = "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"
GITHUB_URL = "https://github.com/anmolvarshney77/SLM-Lending-and-Credit"

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

# ── CELL 0: Title ──────────────────────────────────────────────────────────
cells.append(md(
    "# 🏦 Lending AI — SLM Fine-Tuning (Complete Pipeline)\n",
    "\n",
    "**Hackathon:** Lending AI — Fine-Tune a Small Language Model for Credit Intelligence  \n",
    "**Model:** `meta-llama/Llama-3.2-3B-Instruct` + QLoRA  \n",
    "**GPU Required:** T4 (15 GB) or better  \n",
    "\n",
    "### What this notebook does — end to end:\n",
    "1. ⚙️  Install dependencies + verify GPU\n",
    "2. 📂  Clone repo from GitHub\n",
    "3. 📊  Upload raw dataset → run full data preparation pipeline\n",
    "4. 🤗  Login to HuggingFace\n",
    "5. 🔥  Fine-tune Llama-3.2-3B-Instruct with QLoRA (~20 min)\n",
    "6. 📈  Evaluate: base model vs fine-tuned (accuracy, F1, ROUGE)\n",
    "7. 🎯  3 side-by-side demo scenarios\n",
    "8. 💾  Download all outputs\n",
    "\n",
    "> **Just run all cells top to bottom.** `Runtime → Run all`\n",
))

# ── CELL 1: GPU Check ──────────────────────────────────────────────────────
cells.append(md("## ⚙️  Step 1 — Environment Setup"))

cells.append(code(
    "# Verify GPU before anything else\n",
    "import subprocess, sys\n",
    "result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],\n",
    "                        capture_output=True, text=True)\n",
    "if result.returncode == 0:\n",
    "    print('✅ GPU:', result.stdout.strip())\n",
    "else:\n",
    "    print('❌ No GPU detected — go to Runtime → Change runtime type → T4 GPU')\n",
    "    sys.exit()\n",
))

cells.append(code(
    "%%capture\n",
    "!pip install transformers>=4.40.0 peft>=0.10.0 trl>=0.8.6 bitsandbytes>=0.43.0 \\\n",
    "             accelerate>=0.27.0 datasets>=2.18.0 evaluate>=0.4.1 rouge-score \\\n",
    "             scikit-learn openpyxl scipy pyyaml tqdm\n",
    "print('✅ All packages installed')\n",
))

# ── CELL 2: Clone repo ─────────────────────────────────────────────────────
cells.append(md("## 📂  Step 2 — Clone Repository"))

cells.append(code(
    f"!git clone {GITHUB_URL} lending-ai-slm\n",
    "%cd lending-ai-slm\n",
    "!ls\n",
))

# ── CELL 3: Upload dataset ─────────────────────────────────────────────────
cells.append(md(
    "## 📊  Step 3 — Upload Raw Dataset\n",
    "\n",
    "Upload **`Lending_Loan_Portfolio_1000_Raw.xlsx`** when the file picker appears.\n",
))

cells.append(code(
    "import os, shutil\n",
    "from google.colab import files\n",
    "\n",
    "os.makedirs('data/raw', exist_ok=True)\n",
    "print('📁 Select Lending_Loan_Portfolio_1000_Raw.xlsx ...')\n",
    "uploaded = files.upload()\n",
    "for fname in uploaded:\n",
    "    dest = f'data/raw/{fname}'\n",
    "    shutil.move(fname, dest)\n",
    "    print(f'✅ Saved to {dest}')\n",
))

# ── CELL 4: HuggingFace login ──────────────────────────────────────────────
cells.append(md("## 🤗  Step 4 — HuggingFace Login"))

cells.append(code(
    "from huggingface_hub import login\n",
    f'login(token="{HF_TOKEN}", add_to_git_credential=False)\n',
    "print('✅ Logged in to HuggingFace')\n",
))

# ── CELL 5: Run data prep ──────────────────────────────────────────────────
cells.append(md(
    "## 🧹  Step 5 — Data Preparation Pipeline\n",
    "\n",
    "Cleans all 7 dirty fields, engineers FOIR/Credit Utilization/IS_DELINQUENT, "
    "derives Risk and Approval labels, generates 3,000 prompt-completion pairs "
    "(1,000 records × 3 tasks), and exports stratified train/val/test JSONL splits.\n",
))

cells.append(code(
    "import sys\n",
    "sys.path.insert(0, 'src')\n",
    "from data_prep import run_pipeline\n",
    "\n",
    "df = run_pipeline()\n",
    "print('\\n✅ Data preparation complete')\n",
))

# ── CELL 6: Verify data ────────────────────────────────────────────────────
cells.append(md("### Dataset Verification"))

cells.append(code(
    "import json, pandas as pd\n",
    "\n",
    "# Split sizes\n",
    "for split in ['train', 'val', 'test']:\n",
    "    with open(f'data/processed/{split}.jsonl') as f:\n",
    "        n = sum(1 for l in f)\n",
    "    print(f'{split:6s}: {n} examples ({n//3} records × 3 tasks)')\n",
    "\n",
    "print()\n",
    "print('Risk Label distribution:')\n",
    "print(df['RISK_LABEL'].value_counts().to_string())\n",
    "print()\n",
    "print('Approval Label distribution:')\n",
    "print(df['APPROVAL_LABEL'].value_counts().to_string())\n",
))

cells.append(code(
    "# Show one sample prompt-completion pair per task type\n",
    "with open('data/processed/train.jsonl') as f:\n",
    "    examples = [json.loads(l) for l in f]\n",
    "\n",
    "task_keywords = {\n",
    "    'RISK':     ['classify', 'risk category', 'risk assessment'],\n",
    "    'APPROVAL': ['approval', 'approve', 'should this loan'],\n",
    "    'SUMMARY':  ['summarise', 'summarize', 'summary', 'narrative'],\n",
    "}\n",
    "shown = set()\n",
    "for ex in examples:\n",
    "    user = ex['messages'][1]['content'].lower()\n",
    "    for task, kws in task_keywords.items():\n",
    "        if task not in shown and any(k in user for k in kws):\n",
    "            shown.add(task)\n",
    "            print(f'{'='*65}')\n",
    "            print(f'  TASK: {task}')\n",
    "            print(f'{'='*65}')\n",
    "            print(f'USER:\\n{ex[\"messages\"][1][\"content\"][:350]}\\n')\n",
    "            print(f'ASSISTANT:\\n{ex[\"messages\"][2][\"content\"][:500]}')\n",
    "            print()\n",
    "    if len(shown) == 3:\n",
    "        break\n",
))

# ── CELL 7: Load model ─────────────────────────────────────────────────────
cells.append(md(
    "## 🔥  Step 6 — Load Llama-3.2-3B-Instruct with 4-bit Quantization\n",
    "\n",
    "**Why QLoRA?** Full fine-tuning needs ~48 GB VRAM. QLoRA:\n",
    "- Freezes base weights in 4-bit NF4 (Normal Float 4)\n",
    "- Trains only small LoRA adapter matrices (~0.08% of parameters)\n",
    "- Fits in 8 GB VRAM, completes in under 25 minutes\n",
))

cells.append(code(
    "import torch\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
    "\n",
    "MODEL_NAME    = 'meta-llama/Llama-3.2-3B-Instruct'\n",
    "compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_quant_type='nf4',\n",
    "    bnb_4bit_compute_dtype=compute_dtype,\n",
    "    bnb_4bit_use_double_quant=True,\n",
    ")\n",
    "\n",
    "print(f'Loading {MODEL_NAME}...')\n",
    "print('This downloads ~1.5 GB on first run — takes 2-3 minutes')\n",
    "\n",
    "base_model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_NAME,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map='auto',\n",
    "    trust_remote_code=True,\n",
    "    torch_dtype=compute_dtype,\n",
    ")\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\n",
    "tokenizer.pad_token    = tokenizer.eos_token\n",
    "tokenizer.padding_side = 'right'\n",
    "\n",
    "print(f'\\n✅ Model loaded on: {next(base_model.parameters()).device}')\n",
    "print(f'   Vocab size: {tokenizer.vocab_size:,}')\n",
    "print(f'   Compute dtype: {compute_dtype}')\n",
))

# ── CELL 8: Apply LoRA ─────────────────────────────────────────────────────
cells.append(md(
    "### Apply LoRA Adapters\n",
    "\n",
    "| Param | Value | Rationale |\n",
    "|-------|-------|----------|\n",
    "| `r` | 16 | Enough capacity for domain adaptation without overfitting 2,400 examples |\n",
    "| `alpha` | 32 | Standard 2× scaling — stable training |\n",
    "| `dropout` | 0.05 | Light regularization |\n",
    "| `target_modules` | q,k,v,o projections | Adapt all attention heads for best domain transfer |\n",
))

cells.append(code(
    "base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)\n",
    "base_model.config.use_cache = False\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    lora_dropout=0.05,\n",
    "    bias='none',\n",
    "    task_type='CAUSAL_LM',\n",
    "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n",
    ")\n",
    "\n",
    "model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "trainable, total = model.get_nb_trainable_parameters()\n",
    "print(f'Trainable params: {trainable:,} ({100*trainable/total:.4f}% of {total:,})')\n",
    "print('✅ Base model weights are FROZEN. Only LoRA adapters will update.')\n",
))

# ── CELL 9: Train ──────────────────────────────────────────────────────────
cells.append(md(
    "## 🚂  Step 7 — Fine-Tune with SFTTrainer\n",
    "\n",
    "Expected time: **15–25 minutes** on T4 GPU.  \n",
    "Watch for `eval_loss` decreasing — that confirms the model is learning domain concepts.\n",
))

cells.append(code(
    "import json, time\n",
    "from datasets import Dataset\n",
    "from transformers import TrainingArguments\n",
    "from trl import SFTTrainer\n",
    "\n",
    "def load_jsonl(path):\n",
    "    records = []\n",
    "    with open(path, encoding='utf-8') as f:\n",
    "        for line in f:\n",
    "            line = line.strip()\n",
    "            if line:\n",
    "                records.append(json.loads(line))\n",
    "    return Dataset.from_list(records)\n",
    "\n",
    "train_dataset = load_jsonl('data/processed/train.jsonl')\n",
    "val_dataset   = load_jsonl('data/processed/val.jsonl')\n",
    "print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')\n",
    "\n",
    "use_bf16 = torch.cuda.is_bf16_supported()\n",
    "\n",
    "training_args = TrainingArguments(\n",
    "    output_dir='outputs/adapter',\n",
    "    num_train_epochs=2,\n",
    "    per_device_train_batch_size=4,\n",
    "    per_device_eval_batch_size=4,\n",
    "    gradient_accumulation_steps=4,\n",
    "    gradient_checkpointing=True,\n",
    "    learning_rate=2e-4,\n",
    "    lr_scheduler_type='cosine',\n",
    "    warmup_steps=100,\n",
    "    weight_decay=0.001,\n",
    "    fp16=not use_bf16,\n",
    "    bf16=use_bf16,\n",
    "    evaluation_strategy='steps',\n",
    "    eval_steps=100,\n",
    "    save_strategy='steps',\n",
    "    save_steps=100,\n",
    "    save_total_limit=2,\n",
    "    load_best_model_at_end=True,\n",
    "    metric_for_best_model='eval_loss',\n",
    "    logging_steps=25,\n",
    "    report_to='none',\n",
    "    seed=42,\n",
    ")\n",
    "\n",
    "def formatting_func(example):\n",
    "    return [\n",
    "        tokenizer.apply_chat_template(\n",
    "            ex['messages'], tokenize=False, add_generation_prompt=False\n",
    "        )\n",
    "        for ex in example\n",
    "    ]\n",
    "\n",
    "# `model` is already PEFT-wrapped above — don't pass peft_config here too,\n",
    "# or SFTTrainer will call get_peft_model() again and double-wrap it.\n",
    "trainer = SFTTrainer(\n",
    "    model=model,\n",
    "    tokenizer=tokenizer,\n",
    "    train_dataset=train_dataset,\n",
    "    eval_dataset=val_dataset,\n",
    "    formatting_func=formatting_func,\n",
    "    max_seq_length=512,\n",
    "    args=training_args,\n",
    ")\n",
    "\n",
    "print('Starting training...')\n",
    "print(f'Effective batch size: {4*4} | Steps/epoch: ~{len(train_dataset)//(4*4)}')\n",
    "t0 = time.time()\n",
    "train_result = trainer.train()\n",
    "elapsed = time.time() - t0\n",
    "\n",
    "print(f'\\n✅ Training complete in {elapsed/60:.1f} minutes')\n",
    "print(f'Final train loss: {train_result.metrics.get(\"train_loss\"):.4f}')\n",
))

# ── CELL 10: Plot loss ─────────────────────────────────────────────────────
cells.append(md("### Training Loss Curve"))

cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "\n",
    "log = trainer.state.log_history\n",
    "train_steps  = [x['step'] for x in log if 'loss' in x and 'eval_loss' not in x]\n",
    "train_losses = [x['loss'] for x in log if 'loss' in x and 'eval_loss' not in x]\n",
    "eval_steps   = [x['step'] for x in log if 'eval_loss' in x]\n",
    "eval_losses  = [x['eval_loss'] for x in log if 'eval_loss' in x]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "ax.plot(train_steps,  train_losses, label='Train Loss',      alpha=0.7, color='#3B6EFF')\n",
    "ax.plot(eval_steps,   eval_losses,  label='Validation Loss', alpha=0.9, color='#F59E0B',\n",
    "        linewidth=2, marker='o', markersize=5)\n",
    "ax.set_xlabel('Steps'); ax.set_ylabel('Loss')\n",
    "ax.set_title('QLoRA Fine-Tuning — Loss Curves')\n",
    "ax.legend(); ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('outputs/training_loss.png', dpi=150)\n",
    "plt.show()\n",
    "print('✅ Loss curve saved')\n",
))

# ── CELL 11: Save adapter ──────────────────────────────────────────────────
cells.append(md("### Save LoRA Adapter"))

cells.append(code(
    "import os, json\n",
    "os.makedirs('outputs/adapter', exist_ok=True)\n",
    "\n",
    "trainer.model.save_pretrained('outputs/adapter')\n",
    "tokenizer.save_pretrained('outputs/adapter')\n",
    "\n",
    "metrics = dict(train_result.metrics)\n",
    "metrics['elapsed_minutes'] = elapsed / 60\n",
    "metrics['model_name'] = MODEL_NAME\n",
    "with open('outputs/training_metrics.json', 'w') as f:\n",
    "    json.dump(metrics, f, indent=2, default=str)\n",
    "\n",
    "print('✅ Adapter saved to outputs/adapter/')\n",
    "print('Files:', os.listdir('outputs/adapter'))\n",
))

# ── CELL 12: Evaluation helper functions ───────────────────────────────────
cells.append(md(
    "## 📈  Step 8 — Before vs After Evaluation\n",
    "\n",
    "We evaluate both the **base model** (no fine-tuning) and the **fine-tuned model** "
    "on the same held-out test set.  \n",
    "The adapter is toggled on/off — same weights, one model object.\n",
))

cells.append(code(
    "import re, numpy as np\n",
    "from tqdm import tqdm\n",
    "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
    "\n",
    "DOMAIN_TERMS = [\n",
    "    'bureau', 'dpd', 'foir', 'delinquency', 'delinquent', 'emi',\n",
    "    'credit utilization', 'collection bucket', 'outstanding', 'write-off',\n",
    "    'default', 'sanction', 'repayment', 'overdue', 'risk',\n",
    "]\n",
    "\n",
    "def detect_task(msg):\n",
    "    m = msg.lower()\n",
    "    if any(w in m for w in ['summarise','summarize','summary','narrative']): return 'summary'\n",
    "    if any(w in m for w in ['classify','risk category','risk assessment']):  return 'risk'\n",
    "    if any(w in m for w in ['approval','approve','should this loan']):        return 'approval'\n",
    "    return 'unknown'\n",
    "\n",
    "def extract_label(text, task):\n",
    "    if task == 'risk':\n",
    "        if re.search(r'high risk', text, re.I):   return 'High Risk'\n",
    "        if re.search(r'medium risk', text, re.I): return 'Medium Risk'\n",
    "        if re.search(r'low risk', text, re.I):    return 'Low Risk'\n",
    "        return 'Unknown'\n",
    "    if task == 'approval':\n",
    "        if re.search(r'approve with conditions', text, re.I): return 'Approve with Conditions'\n",
    "        if re.search(r'\\breject\\b', text, re.I):              return 'Reject'\n",
    "        if re.search(r'\\bapprove\\b', text, re.I):             return 'Approve'\n",
    "        return 'Unknown'\n",
    "    return text\n",
    "\n",
    "def domain_recall(text):\n",
    "    t = text.lower()\n",
    "    return sum(1 for w in DOMAIN_TERMS if w in t) / len(DOMAIN_TERMS)\n",
    "\n",
    "@torch.no_grad()\n",
    "def infer(mdl, msgs, max_new=250):\n",
    "    prompt = tokenizer.apply_chat_template(\n",
    "        msgs[:-1], tokenize=False, add_generation_prompt=True\n",
    "    )\n",
    "    inp = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)\n",
    "    inp = {k: v.to(mdl.device) for k, v in inp.items()}\n",
    "    out = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False,\n",
    "                        pad_token_id=tokenizer.pad_token_id)\n",
    "    new = out[0][inp['input_ids'].shape[1]:]\n",
    "    return tokenizer.decode(new, skip_special_tokens=True).strip()\n",
    "\n",
    "with open('data/processed/test.jsonl') as f:\n",
    "    test_data = [json.loads(l) for l in f]\n",
    "print(f'Test examples: {len(test_data)}')\n",
    "print('✅ Evaluation helpers ready')\n",
))

# ── CELL 13: Run eval on both models ──────────────────────────────────────
cells.append(code(
    "from peft import PeftModel\n",
    "\n",
    "def run_eval(mdl, label, cap=33):\n",
    "    r = {'risk':     {'y_true':[], 'y_pred':[], 'dr':[]},\n",
    "         'approval': {'y_true':[], 'y_pred':[]},\n",
    "         'summary':  {'refs':[], 'preds':[], 'dr':[]}}\n",
    "    cnt = {'risk':0,'approval':0,'summary':0}\n",
    "    for ex in tqdm(test_data, desc=label):\n",
    "        msgs = ex['messages']\n",
    "        task = detect_task(msgs[1]['content'])\n",
    "        if task == 'unknown' or cnt.get(task, 0) >= cap: continue\n",
    "        ref  = msgs[2]['content']\n",
    "        pred = infer(mdl, msgs)\n",
    "        if task in ('risk','approval'):\n",
    "            r[task]['y_true'].append(extract_label(ref, task))\n",
    "            r[task]['y_pred'].append(extract_label(pred, task))\n",
    "        if task == 'risk':\n",
    "            r['risk']['dr'].append(domain_recall(pred))\n",
    "        if task == 'summary':\n",
    "            r['summary']['refs'].append(ref)\n",
    "            r['summary']['preds'].append(pred)\n",
    "            r['summary']['dr'].append(domain_recall(pred))\n",
    "        cnt[task] += 1\n",
    "    print(f'{label} — {cnt}')\n",
    "    return r\n",
    "\n",
    "# Disable adapter → base model\n",
    "print('Evaluating BASE model (no fine-tuning)...')\n",
    "model.disable_adapter_layers()\n",
    "base_res = run_eval(model, 'Base Model')\n",
    "\n",
    "# Enable adapter → fine-tuned\n",
    "print('\\nEvaluating FINE-TUNED model...')\n",
    "model.enable_adapter_layers()\n",
    "ft_res = run_eval(model, 'Fine-Tuned')\n",
    "\n",
    "print('\\n✅ Inference complete')\n",
))

# ── CELL 14: Print results table ───────────────────────────────────────────
cells.append(md("### Classification Metrics — Side by Side"))

cells.append(code(
    "import pandas as pd\n",
    "\n",
    "def cls_metrics(y_true, y_pred, labels):\n",
    "    from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
    "    acc = accuracy_score(y_true, y_pred)\n",
    "    f1  = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)\n",
    "    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)\n",
    "    return acc, f1, rep\n",
    "\n",
    "RISK_LABELS = ['Low Risk', 'Medium Risk', 'High Risk']\n",
    "APPR_LABELS = ['Approve', 'Approve with Conditions', 'Reject']\n",
    "\n",
    "ba, bf, br = cls_metrics(base_res['risk']['y_true'],     base_res['risk']['y_pred'],     RISK_LABELS)\n",
    "fa, ff, fr = cls_metrics(ft_res['risk']['y_true'],       ft_res['risk']['y_pred'],       RISK_LABELS)\n",
    "baa,baf,bar= cls_metrics(base_res['approval']['y_true'], base_res['approval']['y_pred'], APPR_LABELS)\n",
    "faa,faf,far= cls_metrics(ft_res['approval']['y_true'],   ft_res['approval']['y_pred'],   APPR_LABELS)\n",
    "\n",
    "print('='*60)\n",
    "print('  CREDIT RISK CLASSIFICATION')\n",
    "print('='*60)\n",
    "rows = []\n",
    "for cls in RISK_LABELS:\n",
    "    b = br.get(cls,{}); f = fr.get(cls,{})\n",
    "    rows.append({'Class': cls,\n",
    "        'Base F1': f\"{b.get('f1-score',0):.3f}\", 'FT F1': f\"{f.get('f1-score',0):.3f}\",\n",
    "        'Base Recall': f\"{b.get('recall',0):.3f}\", 'FT Recall': f\"{f.get('recall',0):.3f}\",\n",
    "    })\n",
    "print(pd.DataFrame(rows).to_string(index=False))\n",
    "print(f'\\nAccuracy: Base {ba:.1%}  →  FT {fa:.1%}  (Δ {fa-ba:+.1%})')\n",
    "print(f'F1 Macro: Base {bf:.4f}  →  FT {ff:.4f}  (Δ {ff-bf:+.4f})')\n",
    "\n",
    "print()\n",
    "print('='*60)\n",
    "print('  LOAN APPROVAL RECOMMENDATION')\n",
    "print('='*60)\n",
    "rows2 = []\n",
    "for cls in APPR_LABELS:\n",
    "    b = bar.get(cls,{}); f = far.get(cls,{})\n",
    "    rows2.append({'Class': cls,\n",
    "        'Base F1': f\"{b.get('f1-score',0):.3f}\", 'FT F1': f\"{f.get('f1-score',0):.3f}\",\n",
    "    })\n",
    "print(pd.DataFrame(rows2).to_string(index=False))\n",
    "print(f'\\nAccuracy: Base {baa:.1%}  →  FT {faa:.1%}  (Δ {faa-baa:+.1%})')\n",
    "print(f'F1 Macro: Base {baf:.4f}  →  FT {faf:.4f}  (Δ {faf-baf:+.4f})')\n",
))

# ── CELL 15: Business impact ───────────────────────────────────────────────
cells.append(md(
    "### 🎯 Business Impact — High Risk Borrower Detection\n",
    "\n",
    "The most important metric: what fraction of high-risk borrowers (potential defaulters) "
    "does each model correctly identify?\n",
))

cells.append(code(
    "y_true_risk = base_res['risk']['y_true']\n",
    "b_pred_risk  = base_res['risk']['y_pred']\n",
    "f_pred_risk  = ft_res['risk']['y_pred']\n",
    "\n",
    "n_high = sum(t == 'High Risk' for t in y_true_risk)\n",
    "b_tp   = sum(p == 'High Risk' and t == 'High Risk' for p,t in zip(b_pred_risk, y_true_risk))\n",
    "f_tp   = sum(p == 'High Risk' and t == 'High Risk' for p,t in zip(f_pred_risk, y_true_risk))\n",
    "\n",
    "b_recall = b_tp / n_high if n_high else 0\n",
    "f_recall = f_tp / n_high if n_high else 0\n",
    "\n",
    "# Domain term recall\n",
    "b_dr = np.mean(base_res['risk']['dr']) if base_res['risk']['dr'] else 0\n",
    "f_dr = np.mean(ft_res['risk']['dr'])   if ft_res['risk']['dr']   else 0\n",
    "\n",
    "print('HIGH-RISK BORROWER RECALL')\n",
    "print(f'  High Risk borrowers in test set: {n_high}')\n",
    "print(f'  Base model correctly flagged:    {b_recall:.1%}')\n",
    "print(f'  Fine-tuned correctly flagged:    {f_recall:.1%}')\n",
    "print(f'  Improvement:                     {f_recall-b_recall:+.1%}')\n",
    "print()\n",
    "print('DOMAIN TERM RECALL (Risk task)')\n",
    "print(f'  Base model:  {b_dr:.1%} of domain terms used correctly')\n",
    "print(f'  Fine-tuned:  {f_dr:.1%} of domain terms used correctly')\n",
    "print(f'  Improvement: {f_dr-b_dr:+.1%}')\n",
))

# ── CELL 16: ROUGE ─────────────────────────────────────────────────────────
cells.append(md("### Loan Summary ROUGE Scores"))

cells.append(code(
    "try:\n",
    "    import evaluate as hf_eval\n",
    "    rouge = hf_eval.load('rouge')\n",
    "    if base_res['summary']['preds'] and ft_res['summary']['preds']:\n",
    "        br = rouge.compute(predictions=base_res['summary']['preds'], references=base_res['summary']['refs'])\n",
    "        fr = rouge.compute(predictions=ft_res['summary']['preds'],   references=ft_res['summary']['refs'])\n",
    "        bd = np.mean(base_res['summary']['dr']) if base_res['summary']['dr'] else 0\n",
    "        fd = np.mean(ft_res['summary']['dr'])   if ft_res['summary']['dr']   else 0\n",
    "        print('ROUGE SCORES')\n",
    "        for k in ['rouge1','rouge2','rougeL']:\n",
    "            print(f'  {k}: Base {br[k]:.4f}  →  FT {fr[k]:.4f}  (Δ {fr[k]-br[k]:+.4f})')\n",
    "        print(f'  Domain Recall: Base {bd:.1%}  →  FT {fd:.1%}  (Δ {fd-bd:+.1%})')\n",
    "    else:\n",
    "        print('No summary examples in test sample — increase cap_per_task')\n",
    "except Exception as e:\n",
    "    print(f'ROUGE skipped: {e}')\n",
))

# ── CELL 17: Demo scenarios ────────────────────────────────────────────────
cells.append(md(
    "## 🎬  Step 9 — Three Demo Scenarios: Base vs Fine-Tuned\n",
    "\n",
    "These are the three representative scenarios the judges will see.  \n",
    "The contrast demonstrates what domain fine-tuning actually buys.\n",
))

SCENARIOS = [
    {
        "title": "Scenario 1 — Low-Risk Salaried Borrower (Expected: Low Risk / Approve)",
        "user": (
            "Classify the credit risk for the following borrower. "
            "Respond with Low Risk, Medium Risk, or High Risk, followed by a detailed justification.\n\n"
            "Borrower Profile:\n"
            "  Age: 35 | Gender: Male | State: Maharashtra\n"
            "  Occupation: Salaried | Monthly Income: ₹75,000\n"
            "Loan Details:\n"
            "  Product: Personal Loan | Sanctioned: ₹4,50,000\n"
            "  EMI: ₹11,200 | Outstanding: ₹2,80,000\n"
            "Credit & Repayment:\n"
            "  Bureau Score: 792 (Very Good, 750-799)\n"
            "  Current DPD: 0 | Max DPD: 0 | Collection Bucket: Current\n"
            "  Default Flag: No | Write-Off Flag: No\n"
            "Derived Metrics:\n"
            "  FOIR: 0.15 (14.9%) | Credit Utilization: 62.2%"
        ),
    },
    {
        "title": "Scenario 2 — High-Risk Delinquent Borrower (Expected: High Risk / Reject)",
        "user": (
            "Provide a loan approval recommendation for this application. "
            "Choose from: Approve, Approve with Conditions, or Reject.\n\n"
            "Borrower Profile:\n"
            "  Age: 42 | Gender: Male | State: Delhi\n"
            "  Occupation: Self Employed | Monthly Income: ₹42,000\n"
            "Loan Details:\n"
            "  Product: Personal Loan | Sanctioned: ₹3,00,000\n"
            "  EMI: ₹28,560 | Outstanding: ₹2,45,000\n"
            "Credit & Repayment:\n"
            "  Bureau Score: 624 (High Risk, <650)\n"
            "  Current DPD: 75 | Max DPD: 90 | Collection Bucket: 61-90\n"
            "  Default Flag: No | Write-Off Flag: No\n"
            "Derived Metrics:\n"
            "  FOIR: 0.68 (68.0%) | Credit Utilization: 81.7%"
        ),
    },
    {
        "title": "Scenario 3 — Borderline Medium-Risk (Expected: Approve with Conditions)",
        "user": (
            "Summarise this borrower's loan profile for an underwriter review.\n\n"
            "Borrower Profile:\n"
            "  Age: 29 | Gender: Female | State: Rajasthan\n"
            "  Occupation: Salaried | Monthly Income: ₹58,000\n"
            "Loan Details:\n"
            "  Product: Vehicle Loan | Sanctioned: ₹5,00,000\n"
            "  EMI: ₹13,200 | Outstanding: ₹3,90,000\n"
            "Credit & Repayment:\n"
            "  Bureau Score: 712 (Good, 700-749)\n"
            "  Current DPD: 0 | Max DPD: 45 | Collection Bucket: Current\n"
            "  Default Flag: No | Write-Off Flag: No\n"
            "Derived Metrics:\n"
            "  FOIR: 0.51 (51.0%) | Credit Utilization: 78.0%"
        ),
    },
]

SYSTEM = (
    "You are a Lending Intelligence Assistant at ABC Finance Ltd. "
    "You understand domain concepts: DPD, FOIR, Bureau Score tiers, Collection Buckets, "
    "Credit Utilization. Provide precise, actionable assessments grounded in lending definitions."
)

for i, sc in enumerate(SCENARIOS, 1):
    cells.append(md(f"### {sc['title']}"))
    user_escaped = sc['user'].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    cells.append(code(
        f"scenario_{i}_msgs = [\n",
        f'    {{"role":"system","content":"{SYSTEM}"}},\n',
        f'    {{"role":"user",  "content":"""\n{sc["user"]}\n"""}},\n',
        f"]\n",
        f"\n",
        f"model.disable_adapter_layers()\n",
        f"base_ans_{i} = infer(model, scenario_{i}_msgs, max_new=300)\n",
        f"\n",
        f"model.enable_adapter_layers()\n",
        f"ft_ans_{i}   = infer(model, scenario_{i}_msgs, max_new=300)\n",
        f"\n",
        f"sep = '─' * 60\n",
        f"print(sep)\n",
        f"print('  BASE MODEL (no fine-tuning):')\n",
        f"print(sep)\n",
        f"print(base_ans_{i})\n",
        f"print()\n",
        f"print(sep)\n",
        f"print('  FINE-TUNED MODEL:')\n",
        f"print(sep)\n",
        f"print(ft_ans_{i})\n",
    ))

# ── CELL 18: Save full eval report ─────────────────────────────────────────
cells.append(md("## 💾  Step 10 — Save & Download All Outputs"))

cells.append(code(
    "import os, json\n",
    "os.makedirs('outputs', exist_ok=True)\n",
    "\n",
    "# Build evaluation report markdown\n",
    "report_lines = [\n",
    "    '# Lending AI SLM — Evaluation Report',\n",
    "    '',\n",
    "    '## Credit Risk Classification',\n",
    "    f'| Metric | Base | Fine-Tuned | Delta |',\n",
    "    f'|--------|------|------------|-------|',\n",
    "    f'| Accuracy | {ba:.1%} | {fa:.1%} | {fa-ba:+.1%} |',\n",
    "    f'| F1 Macro | {bf:.4f} | {ff:.4f} | {ff-bf:+.4f} |',\n",
    "    '',\n",
    "    '## Loan Approval Recommendation',\n",
    "    f'| Metric | Base | Fine-Tuned | Delta |',\n",
    "    f'|--------|------|------------|-------|',\n",
    "    f'| Accuracy | {baa:.1%} | {faa:.1%} | {faa-baa:+.1%} |',\n",
    "    f'| F1 Macro | {baf:.4f} | {faf:.4f} | {faf-baf:+.4f} |',\n",
    "    '',\n",
    "    '## Business Impact',\n",
    "    f'- High Risk Recall — Base: {b_recall:.1%} → Fine-Tuned: {f_recall:.1%} (Δ {f_recall-b_recall:+.1%})',\n",
    "    f'- Domain Term Recall — Base: {b_dr:.1%} → Fine-Tuned: {f_dr:.1%}',\n",
    "]\n",
    "\n",
    "with open('outputs/evaluation_report.md', 'w') as f:\n",
    "    f.write('\\n'.join(report_lines))\n",
    "\n",
    "eval_data = {\n",
    "    'risk_accuracy_base': ba, 'risk_accuracy_ft': fa,\n",
    "    'risk_f1_macro_base': bf, 'risk_f1_macro_ft': ff,\n",
    "    'approval_accuracy_base': baa, 'approval_accuracy_ft': faa,\n",
    "    'high_risk_recall_base': b_recall, 'high_risk_recall_ft': f_recall,\n",
    "    'domain_recall_base': b_dr, 'domain_recall_ft': f_dr,\n",
    "}\n",
    "with open('outputs/evaluation_results.json', 'w') as f:\n",
    "    json.dump(eval_data, f, indent=2)\n",
    "\n",
    "print('✅ Reports saved')\n",
))

cells.append(code(
    "import shutil\n",
    "from google.colab import files\n",
    "\n",
    "# Zip adapter weights\n",
    "shutil.make_archive('adapter_weights', 'zip', 'outputs/adapter')\n",
    "\n",
    "# Zip all outputs\n",
    "shutil.make_archive('all_outputs', 'zip', 'outputs')\n",
    "\n",
    "print('Downloading files...')\n",
    "files.download('adapter_weights.zip')       # LoRA adapter (~60 MB)\n",
    "files.download('all_outputs.zip')           # All reports + loss curve\n",
    "files.download('outputs/evaluation_report.md')\n",
    "files.download('outputs/training_metrics.json')\n",
    "print('✅ All files downloaded')\n",
))

# ── CELL 19: Push outputs to GitHub ───────────────────────────────────────
cells.append(md(
    "## 📤  Step 11 — Push Results to GitHub\n",
    "\n",
    "Commits the JSONL datasets, outputs, and training metrics to your repo.\n",
))

cells.append(code(
    "import subprocess\n",
    "\n",
    "# Copy generated files back to repo\n",
    "!cp -r data/processed outputs notebooks .\n",
    "\n",
    "!git config user.email 'anmol@lyzr.ai'\n",
    "!git config user.name 'Anmol Varshney'\n",
    "\n",
    "!git add data/processed/ outputs/ outputs/evaluation_report.md outputs/training_metrics.json\n",
    "!git add --force outputs/evaluation_report.md outputs/training_metrics.json\n",
    "\n",
    "!git commit -m 'Add training outputs, JSONL datasets, and evaluation report'\n",
    "\n",
    f"!git remote set-url origin https://anmolvarshney77:{HF_TOKEN}@github.com/anmolvarshney77/SLM-Lending-and-Credit.git\n",
    "\n",
    "# Note: above uses HF token as placeholder — replace with your GitHub PAT below\n",
    "# Create a GitHub PAT at: https://github.com/settings/tokens\n",
    "GITHUB_PAT = 'YOUR_GITHUB_PAT_HERE'   # <-- replace this\n",
    "!git remote set-url origin https://anmolvarshney77:{GITHUB_PAT}@github.com/anmolvarshney77/SLM-Lending-and-Credit.git\n",
    "!git push origin main\n",
    "print('✅ Pushed to GitHub')\n",
))

# ── FINAL: Checklist ───────────────────────────────────────────────────────
cells.append(md(
    "---\n",
    "\n",
    "## ✅  Submission Checklist\n",
    "\n",
    "- [ ] `data/processed/train.jsonl` — 2,400 training examples\n",
    "- [ ] `data/processed/val.jsonl` — 300 validation examples  \n",
    "- [ ] `data/processed/test.jsonl` — 300 held-out test examples\n",
    "- [ ] `outputs/adapter/` — LoRA adapter weights\n",
    "- [ ] `outputs/training_metrics.json` — loss curves\n",
    "- [ ] `outputs/evaluation_report.md` — before-vs-after comparison\n",
    "- [ ] `notebooks/01_data_preparation.ipynb` — data pipeline walkthrough\n",
    "- [ ] `notebooks/02_fine_tuning.ipynb` — training walkthrough\n",
    "- [ ] `notebooks/03_evaluation.ipynb` — evaluation walkthrough\n",
    "- [ ] `README.md` — setup and run instructions\n",
    "- [ ] Add `azentio-talent-Aquisition` as GitHub collaborator\n",
    "- [ ] Email repo URL to tateam@azentio.com\n",
))

# ── Build notebook dict ────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_path = "notebooks/lending_ai_colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅  Notebook written to {out_path}")
print(f"    Cells: {len(cells)}")
