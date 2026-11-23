"""Réglages partagés par tous les scripts du projet."""

import os
from pathlib import Path

from dotenv import load_dotenv

RACINE = Path(__file__).resolve().parent.parent
load_dotenv(RACINE / ".env")

# Fichiers
ARTICLES_FILE = RACINE / "data" / "processed" / "articles.jsonl"
CHUNKS_FILE = RACINE / "data" / "processed" / "chunks.jsonl"

# Base de données
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

# Modèle d'embeddings.
# Leçon de l'étape 1 : un modèle de paraphrase rate les questions reformulées.
# multilingual-e5 est entraîné pour la RECHERCHE (question -> passage).
# Il attend un préfixe : "query: " pour une question, "passage: " pour un passage.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384  # taille des vecteurs produits par ce modèle

# Découpage : un passage fait au plus ~1 500 caractères (~350 tokens).
MAX_CHARS_CHUNK = 1500

# LLM (étape 4) : Groq, une API compatible OpenAI, gratuite avec limites.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Le nom du modèle change avec le temps : vérifier la liste avec
#   python src/lister_modeles_groq.py
# et le modifier dans le fichier .env si besoin.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Nombre de passages envoyés au LLM
TOP_K = 5
