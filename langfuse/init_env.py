"""
Génère langfuse/.env (non versionné) : secrets aléatoires du serveur Langfuse local, et
organisation, projet et clés d'API créés au premier démarrage.

Usage : python langfuse/init_env.py   (n'écrase pas un .env existant)
Puis reporter les deux clés affichées dans le .env du projet (LANGFUSE_PUBLIC_KEY / _SECRET_KEY).
"""
import secrets
import sys
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def main():
    if ENV_PATH.exists():
        sys.exit(f"{ENV_PATH} existe déjà : le supprimer d'abord pour régénérer les secrets "
                 "(les données déjà stockées deviendraient illisibles).")
    pg, minio = secrets.token_hex(16), secrets.token_hex(16)
    public_key, secret_key = f"pk-lf-{secrets.token_hex(16)}", f"sk-lf-{secrets.token_hex(16)}"
    values = {
        "SALT": secrets.token_hex(16),
        "ENCRYPTION_KEY": secrets.token_hex(32),
        "NEXTAUTH_SECRET": secrets.token_hex(32),
        "TELEMETRY_ENABLED": "false",
        "POSTGRES_PASSWORD": pg,
        "DATABASE_URL": f"postgresql://postgres:{pg}@postgres:5432/postgres",
        "CLICKHOUSE_PASSWORD": secrets.token_hex(16),
        "REDIS_AUTH": secrets.token_hex(16),
        "MINIO_ROOT_PASSWORD": minio,
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY": minio,
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY": minio,
        "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY": minio,
        "LANGFUSE_INIT_ORG_ID": "rag-genai-eval",
        "LANGFUSE_INIT_ORG_NAME": "RAG GenAI Eval",
        "LANGFUSE_INIT_PROJECT_ID": "assistant-rag",
        "LANGFUSE_INIT_PROJECT_NAME": "Assistant RAG",
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": public_key,
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": secret_key,
    }
    ENV_PATH.write_text("# Généré par init_env.py - ne pas versionner.\n"
                        + "".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    print(f"{ENV_PATH} créé. Clés du projet à reporter dans le .env du projet :")
    print(f"LANGFUSE_PUBLIC_KEY={public_key}\nLANGFUSE_SECRET_KEY={secret_key}\n"
          "LANGFUSE_BASE_URL=http://localhost:3000")


if __name__ == "__main__":
    main()
