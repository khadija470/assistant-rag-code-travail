"""
Étape 5 — Évaluer l'assistant sur un jeu de 20 questions.

Entrée  : data/evaluation/questions.json
          (question, réponse de référence, articles attendus, type)
Sorties : reports/resultats_eval.csv   (une ligne par question, tout le détail)
          reports/resume_eval.md       (le résumé chiffré)

On évalue DEUX choses séparément, car elles échouent pour des raisons différentes :

1. LA RECHERCHE (le "R") — mesure automatique, sans LLM
   - Recall@5        : au moins un article attendu est dans les passages retrouvés
                       (5 passages, ou jusqu'à 8 avec la réécriture de la question)
   - Recall@5 strict : TOUS les articles attendus sont retrouvés (questions à plusieurs articles)

2. LA RÉPONSE (le "G")
   - Citation correcte : le LLM cite au moins un extrait qui vient d'un article attendu
   - Citations invalides : le LLM cite un article absent des extraits fournis
   - Abstention : pour les questions hors corpus, l'assistant doit refuser de répondre
   - Exactitude, complétude, fidélité (0 à 2) : notées par un "LLM juge" qui compare la
     réponse à la référence et aux extraits. Limite connue : le juge est un LLM, il peut
     se tromper (et ici c'est le même modèle que celui qui répond). On vérifie donc
     quelques notes à la main (colonne verif_humaine du CSV).

Lancer :  python src/evaluate.py                    (avec réécriture de la question)
          python src/evaluate.py --sans-reecriture   (sans, pour comparer)
Durée  :  quelques minutes (pauses pour respecter les limites gratuites de Groq).
"""

import argparse
import csv
import json
import re
import time
from collections import defaultdict

import groq

from config import GROQ_API_KEY, GROQ_MODEL, RACINE, REECRITURE_REQUETE
from generation import articles_cites, repondre

QUESTIONS = RACINE / "data" / "evaluation" / "questions.json"
DOSSIER = RACINE / "reports"
PAUSE_ENTRE_QUESTIONS = 3  # secondes

RE_ABSTENTION = re.compile(
    r"ne permettent pas de répondre|ne contiennent pas|aucun(e)? (information|extrait)"
    r"|ne (mentionnent|précisent|traitent) pas|pas d'information",
    re.IGNORECASE,
)

PROMPT_JUGE = """Tu es un évaluateur rigoureux d'un assistant juridique.
On te donne une QUESTION, une RÉPONSE DE RÉFÉRENCE (correcte), les EXTRAITS fournis à
l'assistant et la RÉPONSE DE L'ASSISTANT. Note la réponse de l'assistant :

- exactitude (0, 1 ou 2) : 2 = aucune erreur par rapport à la référence ; 1 = erreur mineure ;
  0 = erreur importante ou réponse fausse.
- completude (0, 1 ou 2) : 2 = contient tous les éléments essentiels de la référence ;
  1 = en oublie une partie ; 0 = l'essentiel manque.
- fidelite (0, 1 ou 2) : 2 = toutes les affirmations figurent dans les extraits ;
  1 = un détail ne figure pas dans les extraits ; 0 = invente des informations.

Réponds UNIQUEMENT avec un objet JSON, sans texte autour :
{"exactitude": 0, "completude": 0, "fidelite": 0, "commentaire": "une phrase"}"""


def appeler_avec_reessai(fonction, *args, **kwargs):
    """Les offres gratuites limitent le nombre d'appels par minute : on attend et on réessaie."""
    for tentative in range(5):
        try:
            return fonction(*args, **kwargs)
        except groq.RateLimitError:
            attente = 20 * (tentative + 1)
            print(f"   Limite Groq atteinte, pause de {attente} s...")
            time.sleep(attente)
    raise RuntimeError("Limite Groq toujours atteinte après 5 essais")


def juger(client: groq.Groq, q: dict, resultat: dict) -> dict:
    extraits = "\n\n".join(
        f"[Art. {s['article']}]\n{s['texte']}" for s in resultat["sources"]
    )
    contenu = (
        f"QUESTION :\n{q['question']}\n\nRÉPONSE DE RÉFÉRENCE :\n{q['reference']}\n\n"
        f"EXTRAITS :\n{extraits}\n\nRÉPONSE DE L'ASSISTANT :\n{resultat['reponse']}"
    )
    completion = appeler_avec_reessai(
        client.chat.completions.create,
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": PROMPT_JUGE},
                  {"role": "user", "content": contenu}],
        temperature=0,
        max_tokens=2000,
    )
    texte = completion.choices[0].message.content or ""
    trouve = re.search(r"\{.*\}", texte, re.DOTALL)
    try:
        return json.loads(trouve.group(0)) if trouve else {}
    except json.JSONDecodeError:
        return {}




def evaluer_question(client: groq.Groq, q: dict, reecrire: bool) -> dict:
    resultat = appeler_avec_reessai(repondre, q["question"], inclure_textes=True, reecrire=reecrire)
    recuperes = [s["article"] for s in resultat["sources"]]
    cites = articles_cites(resultat["reponse"])
    attendus = set(q["articles_attendus"])
    abstention = bool(RE_ABSTENTION.search(resultat["reponse"]))

    ligne = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "question_reecrite": resultat["question_reecrite"] or "",
        "repondable": q["answerable"],
        "articles_attendus": sorted(attendus),
        "articles_recuperes": recuperes,
        "articles_cites": cites,
        "recall5": int(bool(attendus & set(recuperes))) if attendus else "",
        "recall5_strict": int(attendus <= set(recuperes)) if attendus else "",
        "citation_correcte": int(bool(attendus & set(cites))) if attendus else "",
        "citations_invalides": resultat["citations_invalides"],
        "abstention": int(abstention),
        "abstention_correcte": int(abstention != q["answerable"]),
        "exactitude": "", "completude": "", "fidelite": "", "commentaire_juge": "",
        "duree_llm_s": resultat["mesures"]["duree_llm_s"],
        "tokens": (resultat["mesures"]["tokens_entree"] + resultat["mesures"]["tokens_sortie"]
                   + resultat["mesures"]["tokens_reecriture"]),
        "reponse": resultat["reponse"],
        "verif_humaine": "",
    }

    if q["answerable"]:  # on ne note pas le contenu d'une abstention attendue
        notes = juger(client, q, resultat)
        for cle in ("exactitude", "completude", "fidelite"):
            ligne[cle] = notes.get(cle, "")
        ligne["commentaire_juge"] = notes.get("commentaire", "juge illisible")
    return ligne


def moyenne(valeurs) -> str:
    valeurs = [v for v in valeurs if v != "" and v is not None]
    return f"{sum(valeurs) / len(valeurs):.2f}" if valeurs else "—"


def taux(valeurs) -> str:
    valeurs = [v for v in valeurs if v != "" and v is not None]
    return f"{sum(valeurs)}/{len(valeurs)} ({100 * sum(valeurs) / len(valeurs):.0f} %)" if valeurs else "—"


def ecrire_resume(lignes: list[dict], reecrire: bool) -> str:
    rep = [l for l in lignes if l["repondable"]]
    hors = [l for l in lignes if not l["repondable"]]
    par_type = defaultdict(list)
    for l in lignes:
        par_type[l["type"]].append(l)

    md = [
        "# Résultats de l'évaluation",
        "",
        f"Modèle : `{GROQ_MODEL}` — {len(lignes)} questions ({len(rep)} avec réponse dans le Code, "
        f"{len(hors)} hors corpus) — réécriture de la question : **{'oui' if reecrire else 'non'}**.",
        "",
        "## Recherche (retrieval)",
        f"- Recall@5 : **{taux([l['recall5'] for l in rep])}**",
        f"- Recall@5 strict (tous les articles attendus) : {taux([l['recall5_strict'] for l in rep])}",
        "",
        "## Réponses",
        f"- Citation d'un article attendu : **{taux([l['citation_correcte'] for l in rep])}**",
        f"- Réponses citant un article absent des extraits : "
        f"{sum(1 for l in lignes if l['citations_invalides'])}/{len(lignes)}",
        f"- Abstention correcte sur les questions hors corpus : **{taux([l['abstention_correcte'] for l in hors])}**",
        f"- Abstentions inutiles (refus alors que la réponse existe) : {sum(l['abstention'] for l in rep)}/{len(rep)}",
        f"- Notes du LLM juge (sur 2) : exactitude **{moyenne([l['exactitude'] for l in rep])}**, "
        f"complétude **{moyenne([l['completude'] for l in rep])}**, "
        f"fidélité **{moyenne([l['fidelite'] for l in rep])}**",
        f"- Durée moyenne du LLM : {moyenne([l['duree_llm_s'] for l in lignes])} s — "
        f"tokens moyens par question : {moyenne([l['tokens'] for l in lignes])}",
        "",
        "## Par type de question",
        "",
        "| Type | n | Recall@5 | Citation correcte | Exactitude moy. |",
        "|---|---|---|---|---|",
    ]
    for t, ls in par_type.items():
        md.append(f"| {t} | {len(ls)} | {taux([l['recall5'] for l in ls])} | "
                  f"{taux([l['citation_correcte'] for l in ls])} | {moyenne([l['exactitude'] for l in ls])} |")

    echecs = [l for l in lignes if l["recall5"] == 0 or l["citation_correcte"] == 0
              or not l["abstention_correcte"] or l["exactitude"] in (0, 1)]
    md += ["", "## Questions à analyser", ""]
    if not echecs:
        md.append("Aucune.")
    for l in echecs:
        md.append(f"- **{l['id']}** ({l['type']}) : attendus {l['articles_attendus']}, "
                  f"récupérés {l['articles_recuperes']}, cités {l['articles_cites']}, "
                  f"exactitude {l['exactitude']} — {l['commentaire_juge']}")
    md += ["", "Limites : 20 questions seulement ; notes produites par un LLM juge (le même modèle "
           "que l'assistant), à vérifier à la main sur un échantillon (colonne `verif_humaine` du CSV)."]
    return "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer l'assistant RAG")
    parser.add_argument("--sans-reecriture", action="store_true",
                        help="désactiver la réécriture de la question (configuration de départ)")
    args = parser.parse_args()
    reecrire = REECRITURE_REQUETE and not args.sans_reecriture
    suffixe = "_reecriture" if reecrire else ""

    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY manquante dans .env")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    client = groq.Groq(api_key=GROQ_API_KEY)
    DOSSIER.mkdir(exist_ok=True)

    lignes = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']} - {q['question']}")
        ligne = evaluer_question(client, q, reecrire)
        if ligne["question_reecrite"]:
            print(f"   réécrite : {ligne['question_reecrite']}")
        print(f"   recall@5={ligne['recall5']} cites={ligne['articles_cites']} "
              f"abstention={ligne['abstention']} exactitude={ligne['exactitude']}")
        lignes.append(ligne)
        time.sleep(PAUSE_ENTRE_QUESTIONS)

    with (DOSSIER / f"resultats_eval{suffixe}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(lignes)

    resume = ecrire_resume(lignes, reecrire)
    (DOSSIER / f"resume_eval{suffixe}.md").write_text(resume, encoding="utf-8")
    print("\n" + resume)
    print(f"Détail : reports/resultats_eval{suffixe}.csv — Résumé : reports/resume_eval{suffixe}.md")


if __name__ == "__main__":
    main()
