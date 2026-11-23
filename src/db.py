"""Connexion à PostgreSQL et création de la table des passages (avec pgvector)."""

import psycopg
from pgvector.psycopg import register_vector

from config import DATABASE_URL, EMBEDDING_DIM


def connecter() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # permet d'envoyer/recevoir des vecteurs numpy
    return conn


def creer_schema(conn: psycopg.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS passages (
            id            TEXT PRIMARY KEY,
            article_id    TEXT NOT NULL,
            numero        INTEGER NOT NULL,
            titre_article TEXT NOT NULL,
            contexte      TEXT,
            citation      TEXT NOT NULL,
            page_jo       INTEGER,
            partie        TEXT,
            texte         TEXT NOT NULL,
            embedding     vector({EMBEDDING_DIM}) NOT NULL
        )
    """)
    # Index HNSW : accélère la recherche des vecteurs les plus proches
    # (distance cosinus). Indispensable quand le corpus grandit.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS passages_embedding_idx
        ON passages USING hnsw (embedding vector_cosine_ops)
    """)
    conn.commit()
