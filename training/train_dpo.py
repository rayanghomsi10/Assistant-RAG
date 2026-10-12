"""
Étape DPO 3/4 : entraînement DPO + QLoRA (GPU 16 Go : Colab / Kaggle, voir dpo_colab.ipynb).

- Modèle de base : Llama 3.1 8B Instruct, le même que le générateur Ollama (llama3.1:8b), pour
  que l'adaptateur s'applique au modèle réellement déployé.
- QLoRA : modèle chargé en 4 bits (NF4), seuls des adaptateurs LoRA sont entraînés. Avec PEFT,
  le modèle de référence de DPO est le même modèle adaptateurs désactivés : pas de seconde copie
  en mémoire.
- Gabarit de chat : celui d'Ollama pour Llama 3.1 (sans les lignes de date que le gabarit
  Hugging Face ajoute au message système), pour entraîner sur le format exact de l'inférence.
- Les exemples plus longs que --max-length sont écartés plutôt que tronqués (une troncature
  couperait la fin de la réponse, c'est-à-dire précisément ce qu'on optimise).

Usage :
    python training/train_dpo.py --data training/data/preferences.jsonl
Sortie : training/outputs/dpo-lora/ (adaptateur LoRA + tokenizer + training_summary.json)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Format des prompts envoyés par Ollama au modèle llama3.1 (sans outils).
OLLAMA_LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
    "{{- message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_pairs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"],
             "id": r.get("id", str(i)), "type": r.get("type", "")} for i, r in enumerate(rows)]


def sequence_length(tokenizer, example: dict) -> int:
    """Longueur (en tokens) du prompt suivi de la plus longue des deux réponses."""
    longest = max(example["chosen"], example["rejected"], key=lambda m: len(m[0]["content"]))
    ids = tokenizer.apply_chat_template(example["prompt"] + longest, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):   # BatchEncoding selon les versions
        ids = ids["input_ids"]
    return len(ids)


def split_pairs(pairs: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * eval_frac)) if eval_frac > 0 and len(pairs) > 10 else 0
    return pairs[n_eval:], pairs[:n_eval]


def prepare_tokenizer(tokenizer):
    tokenizer.chat_template = OLLAMA_LLAMA3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_trainer(model, tokenizer, train_rows: list[dict], eval_rows: list[dict], output_dir: str,
                  args, quantization_config=None, peft_config=None, extra_config: dict | None = None):
    """Assemble le DPOTrainer. `model` peut être un identifiant Hugging Face (chargé par TRL avec
    `quantization_config`) ou un modèle déjà instancié (tests)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    columns = ["prompt", "chosen", "rejected"]
    train_ds = Dataset.from_list([{k: r[k] for k in columns} for r in train_rows])
    eval_ds = Dataset.from_list([{k: r[k] for k in columns} for r in eval_rows]) if eval_rows else None

    config = DPOConfig(
        output_dir=output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,        # transformers >= 5 : un flottant < 1 est un ratio du nombre de pas
        logging_steps=5,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        **(extra_config or {}),
    )
    return DPOTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                      processing_class=tokenizer, quantization_config=quantization_config,
                      peft_config=peft_config)


def summarize_logs(log_history: list[dict]) -> dict:
    """Dernières valeurs des métriques DPO utiles (marges et exactitude des récompenses implicites)."""
    keys = ["loss", "rewards/accuracies", "rewards/margins", "eval_loss", "eval_rewards/accuracies",
            "eval_rewards/margins"]
    summary = {}
    for entry in log_history:
        for k in keys:
            if k in entry:
                summary[k] = entry[k]
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="training/data/preferences.jsonl")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output", default="training/outputs/dpo-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("Un GPU CUDA est nécessaire (Colab/Kaggle : Exécution > Modifier le type d'exécution > GPU).")
    # bf16 natif seulement à partir d'Ampere (capacité 8.x) ; T4 (7.5) -> fp16. On n'utilise pas
    # torch.cuda.is_bf16_supported(), qui peut répondre oui sur T4 via une émulation lente.
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(args.base_model))
    pairs = load_pairs(Path(args.data))
    kept = [r for r in pairs if sequence_length(tokenizer, r) <= args.max_length]
    print(f"{len(pairs)} paires, {len(pairs) - len(kept)} écartées (> {args.max_length} tokens)")
    train_rows, eval_rows = split_pairs(kept, args.eval_frac, args.seed)
    print(f"entraînement : {len(train_rows)}  validation : {len(eval_rows)}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM")
    trainer = build_trainer(
        args.base_model, tokenizer, train_rows, eval_rows, args.output, args,
        quantization_config=quantization_config, peft_config=peft_config,
        extra_config={
            "bf16": bf16, "fp16": not bf16,
            "gradient_checkpointing": True,
            "model_init_kwargs": {"dtype": compute_dtype, "device_map": {"": 0}},
        },
    )
    trainer.train()
    trainer.save_model(args.output)          # adaptateur LoRA (safetensors)
    tokenizer.save_pretrained(args.output)

    summary = {
        "base_model": args.base_model, "n_train": len(train_rows), "n_eval": len(eval_rows),
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data", "output")},
        "final_metrics": summarize_logs(trainer.state.log_history),
    }
    Path(args.output, "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["final_metrics"], indent=2))
    print(f"Adaptateur enregistré dans {args.output}. Étape suivante : training/export_ollama.py")


if __name__ == "__main__":
    main()
