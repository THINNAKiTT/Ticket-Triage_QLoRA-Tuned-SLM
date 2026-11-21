"""Merge the LoRA adapter into the base weights for adapter-free deployment."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen2.5-0.5b-triage-lora")
    p.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-triage-merged")
    args = p.parse_args()

    print("Loading base model in bf16 on CPU to merge weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")

    model = PeftModel.from_pretrained(base_model, args.adapter)
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
