"""
Étape DPO 4/4 : rendre le modèle entraîné utilisable par Ollama, puis l'évaluer.

Deux voies :

1. Adaptateur (par défaut, rapide) : Ollama applique l'adaptateur LoRA au modèle de base
   llama3.1:8b (ADAPTER dans le Modelfile). Quelques dizaines de Mo à copier, rien à fusionner.
   Limite documentée par Ollama : l'adaptateur a été entraîné sur une base quantifiée en NF4
   (QLoRA) et s'applique à une base quantifiée différemment (Q4_K_M) ; le résultat peut s'écarter
   de celui obtenu à l'entraînement. L'évaluation tranche.

2. Fusion (--merge, plus lourd) : l'adaptateur est fusionné dans les poids fp16 du modèle de base
   (~16 Go, prévoir ~32 Go de RAM : Kaggle convient, Colab gratuit non), puis Ollama importe et
   quantifie le résultat (`ollama create --quantize q4_K_M`). À utiliser si la voie 1 dégrade
   les résultats.

Usage :
    python training/export_ollama.py --adapter training/outputs/dpo-lora
    ollama create rag-assistant-dpo -f training/outputs/ollama/Modelfile
    python evaluation/run_eval.py --run-name dpo_v1 --model rag-assistant-dpo
    python evaluation/compare_runs.py baseline_v2 dpo_v1
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_LLM_MODEL, OLLAMA_NUM_CTX
from training.train_dpo import DEFAULT_BASE_MODEL

# Gabarit Ollama de Llama 3.1 (sans outils), nécessaire seulement pour un modèle fusionné importé
# depuis des safetensors ; avec ADAPTER, le gabarit est hérité de llama3.1:8b.
LLAMA3_TEMPLATE = '''{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}<|start_header_id|>assistant<|end_header_id|>

'''

COMMON_PARAMS = f"""PARAMETER num_ctx {OLLAMA_NUM_CTX}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
"""


def adapter_modelfile(base: str) -> str:
    return f"FROM {base}\nADAPTER ./adapter\n{COMMON_PARAMS}"


def merged_modelfile() -> str:
    return f'FROM ./merged\nTEMPLATE """{LLAMA3_TEMPLATE}"""\n{COMMON_PARAMS}'


def export_adapter(adapter_dir: Path, out_dir: Path, base: str) -> Path:
    dest = out_dir / "adapter"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        src = adapter_dir / name
        if not src.exists():
            raise SystemExit(f"{src} introuvable : est-ce bien le dossier de sortie de train_dpo.py ?")
        shutil.copy2(src, dest / name)
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(adapter_modelfile(base), encoding="utf-8")
    return modelfile


def export_merged(adapter_dir: Path, out_dir: Path, base_model: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Chargement de {base_model} en fp16 (CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    merged = out_dir / "merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(merged)
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(merged_modelfile(), encoding="utf-8")
    return modelfile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--adapter", default="training/outputs/dpo-lora")
    p.add_argument("--out", default="training/outputs/ollama")
    p.add_argument("--base", default=OLLAMA_LLM_MODEL, help="Modèle Ollama de base (voie adaptateur)")
    p.add_argument("--merge", action="store_true", help="Fusionner dans les poids fp16 (voie 2)")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Modèle Hugging Face de base (voie 2)")
    p.add_argument("--name", default="rag-assistant-dpo")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        modelfile = export_merged(Path(args.adapter), out_dir, args.base_model)
        create = f"ollama create --quantize q4_K_M {args.name} -f {modelfile}"
    else:
        modelfile = export_adapter(Path(args.adapter), out_dir, args.base)
        create = f"ollama create {args.name} -f {modelfile}"
    print(f"Modelfile écrit : {modelfile}\n\nÉtapes suivantes :\n  {create}\n"
          f"  python evaluation/run_eval.py --run-name dpo_v1 --model {args.name}\n"
          f"  python evaluation/compare_runs.py baseline_v2 dpo_v1")


if __name__ == "__main__":
    main()
