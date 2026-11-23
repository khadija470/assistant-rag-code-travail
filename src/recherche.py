"""
Étape 3c — Rechercher les passages les plus proches d'une question.

C'est le "R" de RAG (Retrieval). La question est transformée en vecteur avec
le MÊME modèle que les passages, puis PostgreSQL renvoie les k passages dont
le vecteur est le plus proche (distance cosinus, opérateur <=> de pgvector).

Lancer :  python src/recherche.py "Quelle est la durée du congé de maternité ?"
"""

import sys
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from db import connecter


@lru_cache(maxsize=1)
def modele() -> SentenceTransformer:
    """Charge le modèle une seule fois, même si on fait plusieurs recherches."""
    return SentenceTransformer(EMBEDDING_MODEL)


def rechercher(question: str, k: int = 5) -> list[dict]:
    # multilingual-e5 : préfixe "query: " pour une question
    vecteur = modele().encode("query: " + question, normalize_embeddings=True)
    with connecter() as conn:
        lignes = conn.execute(
            """SELECT id, numero, titre_article, citation, texte,
                      1 - (embedding <=> %s) AS score
               FROM passages
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (vecteur, vecteur, k),
        ).fetchall()
    colonnes = ["id", "numero", "titre_article", "citation", "texte", "score"]
    return [dict(zip(colonnes, l)) for l in lignes]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Quelle est la durée du congé de maternité ?"
    print(f"Question : {question}\n")
    for rang, p in enumerate(rechercher(question), start=1):
        print(f"{rang}. [{p['score']:.3f}] Article {p['numero']} - {p['titre_article']}")
        print(f"   {p['citation']}")
        print(f"   {p['texte'][:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
