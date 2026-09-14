"""
Étape 7 de la feuille de route : boucle d'évaluation Ragas + MLflow.

Calcule faithfulness, answer_relevancy et context_precision sur le jeu de questions
de référence (evaluation/test_dataset.json), en utilisant le LLM Ollama local comme juge
(pas d'API payante) et le même modèle d'embeddings local que le reste du pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

import _ragas_compat  # noqa: F401 - doit précéder `import ragas` (voir ce module)
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from evaluation.mlflow_tracking import log_eval_run
from ingestion.embeddings import embed_query, embed_texts
from rag_chain.chain import answer_question

TEST_DATASET_PATH = Path(__file__).parent / "test_dataset.json"
RESULTS_PATH = Path(__file__).parent / "last_eval_results.csv"


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Adapte notre modèle d'embeddings local (sentence-transformers) à l'interface
    attendue par Ragas, pour rester sur le même modèle que le reste du pipeline
    plutôt que d'en ajouter un second."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


def load_test_dataset() -> list[dict]:
    with open(TEST_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_dataset(qa_pairs: list[dict]) -> list[dict]:
    """Fait passer chaque question de référence dans la chaîne RAG et collecte les résultats."""
    rows = []
    for i, pair in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] {pair['question']}")
        result = answer_question(pair["question"])
        rows.append({
            "user_input": pair["question"],
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference": pair["ground_truth"],
        })
    return rows


def get_judge_llm() -> LangchainLLMWrapper:
    """LLM juge local via Ollama, plutôt qu'une API payante (voir guide, section E)."""
    chat = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL, temperature=0.0)
    return LangchainLLMWrapper(chat)


def get_judge_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(LocalSentenceTransformerEmbeddings())


def main(run_name: str | None = None) -> dict:
    qa_pairs = load_test_dataset()
    print(f"{len(qa_pairs)} questions de référence chargées.\n")

    print("Génération des réponses via la chaîne RAG ...")
    rows = run_pipeline_on_dataset(qa_pairs)
    dataset = Dataset.from_list(rows)

    print("\nÉvaluation Ragas en cours (LLM juge = Ollama local, ça peut prendre plusieurs minutes)...")
    # Ollama local sert les requêtes en série (OLLAMA_NUM_PARALLEL=1 par défaut). Le max_workers
    # par défaut de Ragas (16) sature la file d'attente et déclenche des TimeoutError avant que
    # les requêtes en fin de file n'aient eu le temps d'être traitées : on limite donc la
    # concurrence et on augmente le budget de temps par requête.
    run_config = RunConfig(timeout=300, max_workers=2, max_retries=3, max_wait=30)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=get_judge_llm(),
        embeddings=get_judge_embeddings(),
        run_config=run_config,
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nRésultats détaillés (par question) sauvegardés dans {RESULTS_PATH}")

    scores = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
    }
    print("\n--- Scores moyens ---")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    log_eval_run(scores, run_name=run_name)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance la boucle d'évaluation Ragas + MLflow")
    parser.add_argument("--run-name", default=None, help="Nom de la run MLflow (ex: 'chunk_500_overlap_50')")
    args = parser.parse_args()
    main(run_name=args.run_name)
