"""
Étape 1 — Découvrir les embeddings.

Objectif : voir de ses propres yeux qu'un modèle d'embeddings transforme
un texte en vecteur de nombres, et que deux textes de SENS proche donnent
des vecteurs proches, même sans mots en commun.

C'est la brique de base de la recherche dans un RAG :
la question de l'utilisateur et les passages du corpus sont transformés
en vecteurs, puis on cherche les passages les plus proches de la question.

Lancer :  python scripts/01_decouvrir_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Modèle multilingue (gère le français), petit et gratuit, tourne sur CPU.
# Il est téléchargé automatiquement la première fois (~470 Mo).
NOM_MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Quelques "passages", comme s'ils venaient du Code du travail.
PASSAGES = [
    "Le contrat de travail à durée déterminée ne peut pas dépasser deux ans.",
    "L'employeur qui rompt le contrat doit respecter un délai de préavis.",
    "Le salarié a droit à un congé payé chaque année.",
    "La durée légale du travail est fixée à 40 heures par semaine.",
    "Le Sénégal est un pays d'Afrique de l'Ouest dont la capitale est Dakar.",
]

# Une question formulée avec d'AUTRES mots que le passage qui y répond.
QUESTION = "Le salarié a-t-il droit à des congés chaque année ?"

def similarite_cosinus(a: np.ndarray, b: np.ndarray) -> float:
    """Mesure à quel point deux vecteurs pointent dans la même direction.
    1 = sens très proche, 0 = aucun rapport."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print(f"Chargement du modèle {NOM_MODELE} ...")
    modele = SentenceTransformer(NOM_MODELE)

    # 1) Transformer un texte en vecteur
    vecteur_question = modele.encode(QUESTION)
    print(f"\nLa question devient un vecteur de {len(vecteur_question)} nombres.")
    print("Les 5 premiers :", np.round(vecteur_question[:5], 3))

    # 2) Transformer tous les passages en vecteurs
    vecteurs_passages = modele.encode(PASSAGES)

    # 3) Comparer la question à chaque passage
    scores = [similarite_cosinus(vecteur_question, v) for v in vecteurs_passages]

    # 4) Classer les passages du plus proche au moins proche (= recherche top-k)
    classement = sorted(zip(scores, PASSAGES), reverse=True)

    print(f"\nQuestion : {QUESTION}\n")
    print("Passages classés par similarité :")
    for rang, (score, passage) in enumerate(classement, start=1):
        print(f"  {rang}. [{score:.3f}] {passage}")

    print(
        "\nÀ observer : le passage sur le « congé payé » devrait arriver premier,"
        "\nalors que la question parle de « jours de vacances » : le modèle compare"
        "\nle SENS, pas les mots. Le passage sur Dakar devrait arriver dernier."
    )


if __name__ == "__main__":
    main()
