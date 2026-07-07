"""
train.py
========
QLoRA fine-tuning of Llama-3.2-3B-Instruct on the lending instruction dataset.

Usage:
    python src/train.py
    python src/train.py --config configs/training_config.yaml

What this script does:
  1. Load and validate the JSONL training and validation datasets
  2. Load Llama-3.2-3B-Instruct with 4-bit NF4 quantization (BitsAndBytes)
  3. Apply LoRA adapters to attention projection layers (PEFT)
  4. Fine-tune with SFTTrainer (TRL) on the instruction-tuning dataset
  5. Log training metrics (loss, eval loss, grad norm) per step
  6. Save the best LoRA adapter checkpoint to outputs/adapter/
  7. Print a training summary report
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU]   {device} — {vram:.1f} GB VRAM")
            return True
        else:
            print("[GPU]   No CUDA GPU detected — training on CPU (very slow, not recommended)")
            return False
    except ImportError:
        print("[Error] PyTorch not installed. Run: pip install torch")
        sys.exit(1)


def load_dataset_from_jsonl(path: str):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def run_training(config: dict):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    print("=" * 60)
    print("  LENDING AI — QLoRA FINE-TUNING")
    print("=" * 60)

    has_gpu = check_gpu()

    # ── 1. Load datasets ─────────────────────────────────────
    train_path = BASE_DIR / config["data"]["train_file"]
    val_path   = BASE_DIR / config["data"]["val_file"]

    print(f"\n[Data]  Loading training data from {train_path}")
    train_dataset = load_dataset_from_jsonl(str(train_path))
    val_dataset   = load_dataset_from_jsonl(str(val_path))
    print(f"[Data]  Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples")

    # ── 2. Configure quantization ────────────────────────────
    q_cfg = config["quantization"]
    compute_dtype = torch.bfloat16 if q_cfg["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16

    # Fall back to float16 if bfloat16 not supported
    if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        compute_dtype = torch.float16
        print("[Warn]  bfloat16 not supported on this GPU — using float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    # ── 3. Load model ────────────────────────────────────────
    model_name = config["model"]["name"]
    print(f"\n[Model] Loading {model_name} with 4-bit quantization...")
    print("[Model] This may take 1–3 minutes on first run (downloading ~1.5 GB)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
    except Exception as e:
        fallback = config["model"].get("fallback", "microsoft/phi-2")
        print(f"[Warn]  Failed to load {model_name}: {e}")
        print(f"[Model] Falling back to {fallback}")
        model = AutoModelForCausalLM.from_pretrained(
            fallback,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        model_name = fallback

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # ── 4. Load tokenizer ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # prevents attention mask issues with causal LM

    # ── 5. Apply LoRA ────────────────────────────────────────
    l_cfg = config["lora"]
    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        task_type=l_cfg["task_type"],
        target_modules=l_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[LoRA]  Trainable parameters: {trainable:,} ({100*trainable/total:.3f}% of {total:,} total)")

    # ── 6. Training arguments ────────────────────────────────
    t_cfg = config["training"]
    output_dir = BASE_DIR / t_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = t_cfg.get("bf16", False) and torch.cuda.is_bf16_supported()
    use_fp16 = t_cfg.get("fp16", False) and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_steps=t_cfg["warmup_steps"],
        weight_decay=t_cfg["weight_decay"],
        fp16=use_fp16,
        bf16=use_bf16,
        evaluation_strategy=t_cfg["evaluation_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        logging_steps=t_cfg["logging_steps"],
        report_to=t_cfg.get("report_to", "none"),
        seed=t_cfg["seed"],
        dataloader_pin_memory=has_gpu,
    )

    # ── 7. SFT Trainer ───────────────────────────────────────
    # We use the tokenizer's chat template to format messages.
    # SFTTrainer with `dataset_text_field` disabled and `formatting_func`
    # set handles the messages list format natively.

    def formatting_func(example):
        """Convert messages list to a single formatted string using chat template."""
        return [
            tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in example
        ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=t_cfg["max_seq_length"],
        args=training_args,
        peft_config=lora_config,
    )

    # ── 8. Train ─────────────────────────────────────────────
    print(f"\n[Train] Starting fine-tuning...")
    print(f"[Train] Effective batch size: {t_cfg['per_device_train_batch_size'] * t_cfg['gradient_accumulation_steps']}")
    print(f"[Train] Epochs: {t_cfg['num_train_epochs']} | Steps per epoch: ~{len(train_dataset) // (t_cfg['per_device_train_batch_size'] * t_cfg['gradient_accumulation_steps'])}")

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # ── 9. Save adapter ──────────────────────────────────────
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n[Save]  LoRA adapter saved to {output_dir}")

    # ── 10. Training summary ─────────────────────────────────
    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total training time:  {elapsed/60:.1f} minutes")
    print(f"  Total steps:          {metrics.get('train_steps', 'N/A')}")
    print(f"  Final training loss:  {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Samples/second:       {metrics.get('train_samples_per_second', 'N/A'):.1f}")
    print(f"  Model saved to:       {output_dir}")
    print("=" * 60)

    # Save metrics to JSON for the evaluation notebook
    metrics["elapsed_minutes"] = elapsed / 60
    metrics["model_name"] = model_name
    metrics_path = BASE_DIR / "outputs" / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Save]  Training metrics saved to {metrics_path}")

    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Lending AI SLM")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "training_config.yaml"),
        help="Path to training_config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
