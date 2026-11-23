"""
Étape 3b — Calculer les embeddings des passages et les stocker dans pgvector.

Entrée : data/processed/chunks.jsonl
Sortie : table "passages" dans PostgreSQL (un vecteur de 384 nombres par passage)

Prérequis : la base tourne (docker compose up -d) et le fichier .env existe.
Lancer depuis la racine du projet :  python src/indexation.py
"""

import json
import time

from sentence_transformers import SentenceTransformer

from config import CHUNKS_FILE, EMBEDDING_MODEL
from db import connecter, creer_schema


def main() -> None:
    chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"{len(chunks)} passages à indexer avec {EMBEDDING_MODEL}")

    modele = SentenceTransformer(EMBEDDING_MODEL)
    debut = time.time()
    # multilingual-e5 : préfixe "passage: " pour les textes du corpus.
    # normalize_embeddings=True : vecteurs de longueur 1, la similarité
    # cosinus devient un simple produit scalaire.
    vecteurs = modele.encode(
        ["passage: " + c["texte_indexe"] for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    print(f"Embeddings calculés en {time.time() - debut:.0f} s")

    with connecter() as conn:
        creer_schema(conn)
        # On repart d'une table vide : relancer le script ne crée pas de doublons.
        conn.execute("TRUNCATE passages")
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO passages (id, article_id, numero, titre_article, contexte,
                                         citation, page_jo, partie, texte, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(c["id"], c["article_id"], c["numero"], c["titre_article"], c["contexte"],
                  c["citation"], c["page_jo"], c["partie"], c["texte"], v)
                 for c, v in zip(chunks, vecteurs)],
            )
        conn.commit()
        total = conn.execute("SELECT count(*) FROM passages").fetchone()[0]

    print(f"{total} passages enregistrés dans la table 'passages'.")


if __name__ == "__main__":
    main()
