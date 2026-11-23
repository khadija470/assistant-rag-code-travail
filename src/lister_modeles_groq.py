"""Affiche les modèles disponibles avec ta clé Groq, pour choisir GROQ_MODEL dans .env.

Lancer :  python src/lister_modeles_groq.py
"""

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)
noms = sorted(m.id for m in client.models.list().data)
print("Modèles disponibles :")
for nom in noms:
    print(("  * " if nom == GROQ_MODEL else "    ") + nom)
print(f"\nModèle configuré dans .env : {GROQ_MODEL}"
      + ("" if GROQ_MODEL in noms else "  <-- INTROUVABLE, choisis-en un dans la liste"))
