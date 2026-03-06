import argparse
import inspect
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_CANDIDATES = (
    "models/Qwen3.5-0.8B",
)

DEFAULT_DATASET_CANDIDATES = (
    "speculative/sft.json",
)

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_model_path(model_path: str | None) -> Path:
    repo_root = resolve_repo_root()
    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()

    for relative_path in DEFAULT_MODEL_CANDIDATES:
        candidate = repo_root / relative_path
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a default model under {repo_root / 'models'}. "
        f"Tried: {searched}. Please pass --model-path explicitly."
    )


def resolve_dataset_path(dataset_path: str | None) -> Path:
    repo_root = resolve_repo_root()
    if dataset_path:
        candidate = Path(dataset_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()

    for relative_path in DEFAULT_DATASET_CANDIDATES:
        candidate = repo_root / relative_path
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a default dataset under {repo_root / 'speculative'}. "
        f"Tried: {searched}. Please pass --dataset-path explicitly."
    )


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def detect_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(record.get("instruction", "") or "").strip()
    user_input = str(record.get("input", "") or "").strip()
    assistant_output = str(
        record.get("output")
        or record.get("response")
        or record.get("completion")
        or ""
    ).strip()

    if not assistant_output:
        raise ValueError("Each sample must contain a non-empty `output` field.")

    messages: list[dict[str, str]] = []
    if instruction:
        messages.append({"role": "system", "content": instruction})

    if user_input:
        messages.append({"role": "user", "content": user_input})
    elif instruction:
        messages.append({"role": "user", "content": "Please answer the request above."})
    else:
        raise ValueError("Each sample must contain `instruction` or `input`.")

    messages.append({"role": "assistant", "content": assistant_output})
    return messages


def format_example(record: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = build_messages(record)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = ""
        for message in messages:
            text += f"{message['role'].upper()}:\n{message['content']}\n\n"
    return {"text": text}


def load_json_dataset(dataset_path: Path, max_samples: int | None) -> Dataset:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def split_dataset(
    dataset: Dataset, eval_ratio: float, seed: int
) -> tuple[Dataset, Dataset | None]:
    if eval_ratio <= 0 or len(dataset) < 2:
        return dataset, None

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    return dataset.map(
        lambda row: format_example(row, tokenizer),
        desc="Formatting dataset with chat template",
    )


def load_model(
    model_path: Path,
    torch_dtype: torch.dtype,
    attn_implementation: str | None,
    trust_remote_code: bool,
):
    common_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        common_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs)
    except (TypeError, ValueError):
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "This model looks multimodal, but your transformers build does not "
                "provide AutoModelForImageTextToText. Upgrade transformers first."
            ) from exc
        model = AutoModelForImageTextToText.from_pretrained(**common_kwargs)

    model.config.use_cache = False
    return model


def build_peft_config(args: argparse.Namespace) -> LoraConfig | None:
    if args.full_finetune:
        return None
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_training_args(args: argparse.Namespace, has_eval: bool):
    common_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
    }

    if args.dtype == "bf16":
        common_kwargs["bf16"] = True
    elif args.dtype == "fp16":
        common_kwargs["fp16"] = True

    sft_signature = inspect.signature(SFTConfig.__init__)
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in sft_signature.parameters
        else "evaluation_strategy"
    )
    if has_eval:
        common_kwargs[strategy_key] = "steps"
        common_kwargs["eval_steps"] = args.eval_steps
    else:
        common_kwargs[strategy_key] = "no"

    if "max_seq_length" in sft_signature.parameters:
        common_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_signature.parameters:
        common_kwargs["max_length"] = args.max_seq_length
    if "dataset_text_field" in sft_signature.parameters:
        common_kwargs["dataset_text_field"] = "text"
    if "packing" in sft_signature.parameters:
        common_kwargs["packing"] = False

    try:
        return SFTConfig(**common_kwargs)
    except TypeError:
        return TrainingArguments(**common_kwargs)


def build_trainer(
    model,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args,
    peft_config: LoraConfig | None,
    max_seq_length: int,
) -> SFTTrainer:
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    if peft_config is not None and "peft_config" in trainer_signature.parameters:
        trainer_kwargs["peft_config"] = peft_config

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_signature.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length
    if "packing" in trainer_signature.parameters:
        trainer_kwargs["packing"] = False

    return SFTTrainer(**trainer_kwargs)


def parse_args() -> argparse.Namespace:
    repo_root = resolve_repo_root()
    parser = argparse.ArgumentParser(
        description="Fine-tune a local Qwen model with TRL SFTTrainer on speculative/sft.json."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the local training json file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path. Defaults to the first existing Qwen candidate under models/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "outputs" / "qwen-sft"),
        help="Directory used to save checkpoints and final adapters.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Optional cap on sample count for debugging.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.02,
        help="Fraction of data reserved for evaluation. Use 0 to disable eval.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Training dtype. `auto` uses bf16 on CUDA, otherwise fp32.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help="Attention backend passed to transformers.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable when the selected model requires custom model code.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Disable LoRA and fine-tune all model weights directly.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma separated LoRA target modules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    args.dataset_path = resolve_dataset_path(args.dataset_path)
    args.model_path = resolve_model_path(args.model_path)
    args.output_dir = resolve_path(args.output_dir, repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    set_seed(args.seed)

    print(f"Dataset: {args.dataset_path}")
    print(f"Model:   {args.model_path}")
    print(f"Output:  {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    raw_dataset = load_json_dataset(args.dataset_path, args.max_samples)
    train_dataset, eval_dataset = split_dataset(raw_dataset, args.eval_ratio, args.seed)
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    if eval_dataset is not None:
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    torch_dtype = detect_torch_dtype(args.dtype)
    model = load_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    peft_config = build_peft_config(args)
    training_args = build_training_args(args, has_eval=eval_dataset is not None)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Training finished. Model artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
