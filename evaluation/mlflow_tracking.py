"""
Suivi d'expérimentations avec MLflow : chaque run d'évaluation logue la configuration
**réellement indexée** (lue dans les métadonnées de l'index Chroma, pas dans le .env du moment),
la configuration de génération, l'empreinte du jeu d'évaluation, le commit git, les scores avec
leurs intervalles de confiance, et les résultats détaillés en artefacts.
"""
import subprocess
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, ROOT_DIR


def git_state() -> dict:
    def _git(*args):
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    try:
        commit = _git("rev-parse", "--short", "HEAD") or "inconnu"
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except FileNotFoundError:
        commit, dirty = "inconnu", False
    return {"git_commit": commit, "git_dirty": str(dirty)}


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_eval_run(params: dict, metrics: dict, artifacts: list[Path], run_name: str | None = None,
                 tags: dict | None = None) -> str:
    """Logue une run d'évaluation. `metrics` : {nom: valeur} (les NaN sont ignorés)."""
    init_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**git_state(), **(tags or {})})
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and v == v})
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        print(f"Run MLflow loguée ({run.info.run_id}). "
              f"Visualiser : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        return run.info.run_id
