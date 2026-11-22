"""
Étape 2 — Extraire les articles du Code du travail depuis le Journal officiel.

Entrée  : data/raw/JO-7931-du-17-septembre-2026.pdf
          (Loi n° 2026-18 du 3 septembre 2026 portant Code du Travail)
Sortie  : data/processed/articles.jsonl  (un article par ligne, au format JSON)

Difficultés traitées :
  1. Le Journal officiel est sur DEUX COLONNES : une extraction naïve mélange
     les colonnes. On lit donc chaque page colonne par colonne.
  2. On retire les en-têtes de page ("JOURNAL OFFICIEL ...") et les numéros.
  3. On recolle les mots coupés en fin de ligne ("syn-\\ndicale" -> "syndicale").
  4. On découpe le texte article par article, en gardant le contexte
     (Titre / Chapitre / Section / Sous-section / Paragraphe) et la page du JO, pour pouvoir CITER.

Lancer depuis la racine du projet :  python src/ingest.py
"""

import json
import re
from pathlib import Path

import pymupdf

RACINE = Path(__file__).resolve().parent.parent
PDF = RACINE / "data" / "raw" / "JO-7931-du-17-septembre-2026.pdf"
SORTIE = RACINE / "data" / "processed" / "articles.jsonl"

SOURCE = {
    "document": "Loi n° 2026-18 du 3 septembre 2026 portant Code du Travail",
    "publication": "Journal officiel de la République du Sénégal n° 7931 (spécial) du 17 septembre 2026",
    "url": "https://www.vie-publique.sn/docs/56984d14-34a4-459e-8820-dbca07d9393c/JO-7931-du-17-septembre-2026.pdf",
}

PREMIERE_PAGE_JO = 1603  # la page 1 du PDF correspond à la page 1603 du JO

# "Article 12. - Titre", "Article premier. - Objet", "Art. 49. - Titre"
RE_ARTICLE = re.compile(r"^(?:Article|Art\.)\s+(premier|\d+)\s*\.?\s*-\s*(.*)$")
# Intitulés : "TITRE V. - ...", "Chapitre II. - ...", "Section première. - ..."
# (sensible à la casse : une phrase comme "au titre de ..." ne doit PAS être prise pour un Titre)
RE_STRUCTURE = re.compile(
    r"^(LIVRE|Livre|TITRE|Titre|CHAPITRE|Chapitre|SECTION|Section|Sous-section|Paragraphe)"
    r"\s+[\wÉé]+\s*\.?\s*-"
)
NIVEAUX = ["livre", "titre", "chapitre", "section", "sous_section", "paragraphe"]


# ------------------------------------------------------------------
# 1. Lire le PDF colonne par colonne
# ------------------------------------------------------------------
def lire_lignes(pdf: Path) -> list[tuple[int, str]]:
    """Retourne la liste des lignes du document, dans l'ordre de lecture,
    avec leur numéro de page du Journal officiel."""
    lignes = []
    doc = pymupdf.open(pdf)
    for index_page, page in enumerate(doc):
        milieu = page.rect.width / 2
        blocs = [b for b in page.get_text("blocks") if b[6] == 0]  # blocs de texte

        # Retirer l'en-tête de page (en haut, contient "JOURNAL OFFICIEL")
        blocs = [b for b in blocs if not (b[1] < 80 and "JOURNAL OFFICIEL" in b[4])]

        # Blocs pleine largeur, colonne de gauche, colonne de droite
        pleine_largeur = [b for b in blocs if b[0] < milieu - 20 and b[2] > milieu + 20]
        reste = [b for b in blocs if b not in pleine_largeur]
        gauche = [b for b in reste if b[0] < milieu]
        droite = [b for b in reste if b[0] >= milieu]

        page_jo = PREMIERE_PAGE_JO + index_page
        for bloc in (sorted(pleine_largeur, key=lambda b: b[1])
                     + sorted(gauche, key=lambda b: b[1])
                     + sorted(droite, key=lambda b: b[1])):
            for ligne in bloc[4].splitlines():
                ligne = ligne.strip()
                if ligne:
                    lignes.append((page_jo, ligne))
    return lignes


# ------------------------------------------------------------------
# 2. Nettoyer le texte
# ------------------------------------------------------------------
def nettoyer(texte: str) -> str:
    texte = re.sub(r"(\w)-\n(\w)", r"\1\2", texte)  # mot coupé en fin de ligne
    texte = re.sub(r"\n(?=[a-zà-ÿ(,;])", " ", texte)   # ligne qui continue une phrase
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{2,}", "\n", texte)
    return texte.strip()


# ------------------------------------------------------------------
# 3. Découper en articles
# ------------------------------------------------------------------
def decouper_articles(lignes: list[tuple[int, str]]) -> list[dict]:
    articles = []
    contexte = {niveau: "" for niveau in NIVEAUX}
    courant = None
    titre_en_cours = False      # le titre d'un article peut tenir sur deux lignes
    structure_en_cours = None   # un intitulé de chapitre aussi

    for page, ligne in lignes:
        m = RE_ARTICLE.match(ligne)
        if m:
            if courant:
                articles.append(courant)
            numero = "1" if m.group(1) == "premier" else m.group(1)
            courant = {
                "numero": int(numero),
                "titre_article": m.group(2).strip(),
                **contexte,
                "page_jo": page,
                "lignes": [],
            }
            titre_en_cours = True
            structure_en_cours = None
            continue

        # Suite d'un intitulé de Titre / Chapitre / Section écrit sur deux lignes
        if structure_en_cours and len(ligne) < 80 and (ligne.isupper() or ligne[0].islower()):
            contexte[structure_en_cours] += " " + ligne
            continue
        structure_en_cours = None

        s = RE_STRUCTURE.match(ligne)
        if s and len(ligne) < 120:
            niveau = s.group(1).lower().replace("-", "_")
            contexte[niveau] = ligne
            # un niveau supérieur remet à zéro les niveaux inférieurs
            for n in NIVEAUX[NIVEAUX.index(niveau) + 1:]:
                contexte[n] = ""
            titre_en_cours = False
            structure_en_cours = niveau
            continue

        if courant is None:
            continue  # tout ce qui précède l'article premier (en-tête du JO)

        # Suite du titre de l'article : ligne courte, en minuscule, sans point final
        if (titre_en_cours and len(ligne) < 60 and ligne[0].islower()
                and not ligne.endswith(".")):
            courant["titre_article"] += " " + ligne
            continue

        titre_en_cours = False
        courant["lignes"].append(ligne)

    if courant:
        articles.append(courant)

    for a in articles:
        # Cas où le titre et le début du texte sont sur la même ligne
        # (ex. article 49) : on coupe au premier mot qui commence une phrase.
        if len(a["titre_article"]) > 110:
            m = re.match(r"^(.{20,}?) ((?:L'|Le |La |Les |Il |Elle |Tout |Toute |Lorsque ).*)$",
                         a["titre_article"])
            if m:
                a["titre_article"] = m.group(1)
                a["lignes"].insert(0, m.group(2))
        a["texte"] = nettoyer("\n".join(a.pop("lignes")))
        a["titre_article"] = re.sub(r"(\w)- (\w)", r"\1\2", a["titre_article"]).strip()
    return articles


def couper_signature(articles: list[dict]) -> None:
    """Le dernier article est suivi de la formule finale ("Fait à Dakar..." / "Fait Dakar...")."""
    dernier = articles[-1]
    m = re.search(r"Fait\s+(à\s+)?Dakar", dernier["texte"])
    if m:
        dernier["texte"] = dernier["texte"][:m.start()].strip()


def main() -> None:
    lignes = lire_lignes(PDF)
    articles = decouper_articles(lignes)
    couper_signature(articles)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("w", encoding="utf-8") as f:
        for a in articles:
            a["id"] = f"ct2026-art-{a['numero']}"
            a["citation"] = f"Code du travail 2026, article {a['numero']} (JO n° 7931, p. {a['page_jo']})"
            a.update(SOURCE)
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Contrôles qualité
    numeros = [a["numero"] for a in articles]
    manquants = sorted(set(range(1, max(numeros) + 1)) - set(numeros))
    doublons = sorted({n for n in numeros if numeros.count(n) > 1})
    vides = [a["numero"] for a in articles if len(a["texte"]) < 20]
    print(f"{len(articles)} articles extraits -> {SORTIE.relative_to(RACINE)}")
    print(f"Articles manquants : {manquants or 'aucun'}")
    print(f"Doublons           : {doublons or 'aucun'}")
    print(f"Articles presque vides (à vérifier) : {vides or 'aucun'}")


if __name__ == "__main__":
    main()
