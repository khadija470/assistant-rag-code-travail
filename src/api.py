"""
Étape 4b — Exposer l'assistant sous forme d'API REST (FastAPI).

Pourquoi une API plutôt qu'un script ?
  - le modèle d'embeddings est chargé UNE SEULE FOIS au démarrage du serveur,
    au lieu d'être rechargé à chaque question (10 à 30 s sous Windows) ;
  - n'importe quelle application (site web, mobile, chatbot) peut l'utiliser ;
  - documentation interactive automatique sur http://localhost:8000/docs

Lancer depuis la racine du projet :
    python src/api.py
puis ouvrir http://localhost:8000/docs dans le navigateur.
"""

from contextlib import asynccontextmanager

import groq
import psycopg
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import TOP_K
from generation import repondre
from recherche import modele


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargé une fois au démarrage : les requêtes suivantes sont rapides.
    print("Chargement du modèle d'embeddings...")
    modele()
    print("Prêt.")
    yield


app = FastAPI(
    title="Assistant RAG - Code du travail du Sénégal (2026)",
    description="Répond aux questions sur la loi n° 2026-18 en citant les articles utilisés.",
    version="0.4.0",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Quelle est la durée du congé de maternité ?"])
    k: int = Field(TOP_K, ge=1, le=10, description="Nombre de passages envoyés au LLM")


@app.get("/sante")
def sante() -> dict:
    return {"statut": "ok"}


@app.post("/question")
def poser_question(q: Question) -> dict:
    try:
        return repondre(q.question, q.k)
    except RuntimeError as erreur:  # ex. clé API manquante
        raise HTTPException(status_code=500, detail=str(erreur))
    except psycopg.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="Base de données injoignable : vérifie que Docker tourne (docker compose up -d).",
        )
    except groq.APIError as erreur:  # clé invalide, quota dépassé, modèle inconnu...
        raise HTTPException(status_code=502, detail=f"Erreur du service LLM (Groq) : {erreur}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
