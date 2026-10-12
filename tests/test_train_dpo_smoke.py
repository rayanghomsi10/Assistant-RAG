"""Smoke test de l'entraînement DPO avec un Llama minuscule initialisé aléatoirement.

Vérifie la plomberie qui dépend de ce projet (format des données, gabarit de chat, tokenisation
par TRL, DPOTrainer + LoRA), pas la qualité. Ignoré si torch / trl / peft ne sont pas installés :
il tourne dans le workflow CI dédié (.github/workflows/training-smoke.yml).

Depuis TRL 1.15, le calcul des log-probabilités de DPO passe par un noyau Triton, qui exige un
GPU CUDA : sur CPU (CI), on vérifie tout jusqu'à l'entraînement ; les pas d'entraînement et la
sauvegarde de l'adaptateur ne sont exécutés que si un GPU est présent (Colab, Kaggle).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from peft import LoraConfig  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from training.train_dpo import build_trainer, parse_args, prepare_tokenizer, sequence_length  # noqa: E402

SPECIALS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "[PAD]", "[UNK]"]
WORDS = "system user assistant contexte question réponse xcom_push airflow dbt je ne trouve pas".split()


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS + list(".,:;?!'[]()_-"))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                   pad_token="[PAD]", unk_token="[UNK]")
    fast.add_special_tokens({"additional_special_tokens": SPECIALS[1:3]})
    return prepare_tokenizer(fast)


def rows(n=8):
    prompt = [{"role": "system", "content": "contexte : airflow xcom_push"},
              {"role": "user", "content": "question airflow ?"}]
    return [{"prompt": prompt,
             "chosen": [{"role": "assistant", "content": "réponse xcom_push [airflow] ."}],
             "rejected": [{"role": "assistant", "content": "je ne trouve pas ."}],
             "id": str(i), "type": "in_scope"} for i in range(n)]


def build(tmp_path):
    tokenizer = tiny_tokenizer()
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id))
    args = parse_args(["--epochs", "1", "--max-length", "128", "--grad-accum", "1", "--batch-size", "2"])
    peft_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    cuda = torch.cuda.is_available()
    trainer = build_trainer(model, tokenizer, rows(8), rows(2), str(tmp_path / "out"), args,
                            peft_config=peft_config,
                            extra_config={"use_cpu": not cuda, "bf16": False, "fp16": False,
                                          "gradient_checkpointing": False, "max_steps": 2})
    return tokenizer, trainer


def test_dpo_trainer_setup(tmp_path):
    tokenizer, trainer = build(tmp_path)
    assert sequence_length(tokenizer, rows(1)[0]) < 64

    # Données tokenisées par TRL avec le gabarit Ollama : le prompt commence par BOS et chaque
    # réponse se termine par <|eot_id|>, le jeton d'arrêt de Llama 3.1 dans Ollama.
    example = trainer.train_dataset[0]
    assert {"prompt_ids", "chosen_ids", "rejected_ids"} <= set(example)
    assert example["prompt_ids"][0] == tokenizer.bos_token_id
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assert example["chosen_ids"][-1] == eot and example["rejected_ids"][-1] == eot
    assert example["chosen_ids"] != example["rejected_ids"]
    assert len(trainer.train_dataset) == 8 and len(trainer.eval_dataset) == 2

    # Seuls les adaptateurs LoRA sont entraînables.
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau Triton de TRL : GPU CUDA requis")
def test_dpo_training_steps(tmp_path):
    _, trainer = build(tmp_path)
    trainer.train()
    trainer.save_model(str(tmp_path / "out"))
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()
    logged = {k for entry in trainer.state.log_history for k in entry}
    assert "train_loss" in logged
