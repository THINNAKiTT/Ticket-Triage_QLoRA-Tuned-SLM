"""QLoRA fine-tune a small Qwen model on bilingual ticket triage + entity extraction."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from plots import plot_loss_curve

RESULTS_DIR = Path("results")
ASSETS_DIR = Path("assets")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-triage-lora")
    p.add_argument("--tag", default="qwen2.5-0.5b", help="name used for results/ and assets/ files")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = load_from_disk("data/train")
    eval_ds = load_from_disk("data/validation")

    # The dataset is in prompt/completion form, so TRL masks the prompt and
    # computes the loss on the assistant JSON only (completion_only_loss=None -> True).
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        activation_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=512,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    n_trainable, n_total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    print("Starting QLoRA fine-tuning...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "seed": args.seed,
        "train_examples": len(train_ds),
        "trainable_params": n_trainable,
        "total_params": n_total,
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "final_train_loss": next((h["train_loss"] for h in reversed(trainer.state.log_history) if "train_loss" in h), None),
        "final_eval_loss": next((h["eval_loss"] for h in reversed(trainer.state.log_history) if "eval_loss" in h), None),
        "log_history": trainer.state.log_history,
    }
    (RESULTS_DIR / f"train_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    plot_loss_curve(summary["log_history"], ASSETS_DIR / f"loss_curve_{args.tag}.png", title=args.model.split("/")[-1])


if __name__ == "__main__":
    main()
