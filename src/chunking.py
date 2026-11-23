"""
Étape 3a — Découper les articles en passages ("chunks").

Entrée  : data/processed/articles.jsonl
Sortie  : data/processed/chunks.jsonl

Règles de découpage, adaptées à un texte de loi :
  - Un article court (la grande majorité) = UN passage. On ne coupe pas un
    article en deux : son sens juridique doit rester entier.
  - Un article long est coupé entre deux paragraphes, jamais au milieu d'une
    phrase, avec un CHEVAUCHEMENT : le dernier paragraphe d'un passage est
    répété au début du suivant, pour ne pas perdre le fil.
  - L'article 3 (Définitions, ~15 000 caractères) est coupé définition par
    définition, regroupées par paquets.
  - Chaque passage reçoit un EN-TÊTE (numéro, titre de l'article, place dans
    le Code). Il est ajouté au texte qui sera transformé en vecteur : un
    passage comme "Il est fixé à dix-huit semaines" n'a de sens qu'avec
    son contexte "Article 277 - Congé de maternité".

Lancer depuis la racine du projet :  python src/chunking.py
"""

import json

from config import ARTICLES_FILE, CHUNKS_FILE, MAX_CHARS_CHUNK

NIVEAUX = ["titre", "chapitre", "section", "sous_section", "paragraphe"]


def contexte_article(article: dict) -> str:
    """Ex. : 'TITRE IX. - ... > Chapitre III. - ... > Section II. - ...'"""
    return " > ".join(article[n] for n in NIVEAUX if article.get(n))


def grouper(paragraphes: list[str], max_chars: int, chevauchement: bool = True) -> list[str]:
    """Regroupe des paragraphes en passages d'au plus max_chars caractères."""
    passages, courant = [], []
    for p in paragraphes:
        if courant and len("\n".join(courant + [p])) > max_chars:
            passages.append("\n".join(courant))
            # chevauchement : on reprend le dernier paragraphe (s'il n'est pas énorme)
            courant = [courant[-1]] if chevauchement and len(courant[-1]) < max_chars // 2 else []
        courant.append(p)
    if courant:
        passages.append("\n".join(courant))
    return passages


def decouper_article(article: dict) -> list[str]:
    texte = article["texte"]
    if len(texte) <= MAX_CHARS_CHUNK:
        return [texte]

    lignes = [l for l in texte.split("\n") if l.strip()]

    if article["numero"] == 3:
        # Définitions : une phrase d'introduction puis des lignes "- terme : définition".
        intro = [l for l in lignes if not l.startswith("- ")]
        definitions = [l for l in lignes if l.startswith("- ")]
        paquets = grouper(definitions, MAX_CHARS_CHUNK, chevauchement=False)
        prefixe = intro[0] + "\n" if intro else ""
        return [prefixe + p for p in paquets]

    return grouper(lignes, MAX_CHARS_CHUNK)


def main() -> None:
    articles = [json.loads(l) for l in ARTICLES_FILE.open(encoding="utf-8")]
    chunks = []

    for article in articles:
        morceaux = decouper_article(article)
        for i, morceau in enumerate(morceaux, start=1):
            entete = f"Article {article['numero']} - {article['titre_article']}"
            contexte = contexte_article(article)
            chunks.append({
                "id": f"{article['id']}-p{i}",
                "article_id": article["id"],
                "numero": article["numero"],
                "titre_article": article["titre_article"],
                "contexte": contexte,
                "citation": article["citation"],
                "page_jo": article["page_jo"],
                "partie": f"{i}/{len(morceaux)}",
                "texte": morceau,
                # Texte réellement transformé en vecteur : en-tête + contexte + contenu
                "texte_indexe": f"{entete}\n{contexte}\n{morceau}",
            })

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    tailles = sorted(len(c["texte"]) for c in chunks)
    coupes = sum(1 for a in articles if len(a["texte"]) > MAX_CHARS_CHUNK)
    print(f"{len(articles)} articles -> {len(chunks)} passages -> {CHUNKS_FILE.name}")
    print(f"Articles coupés en plusieurs passages : {coupes}")
    print(f"Taille des passages (caractères) : min {tailles[0]}, "
          f"médiane {tailles[len(tailles) // 2]}, max {tailles[-1]}")


if __name__ == "__main__":
    main()
