# Assistant RAG — Code du travail sénégalais

Assistant qui répond à des questions sur le **nouveau Code du travail sénégalais**
(loi n° 2026-18 du 3 septembre 2026, en vigueur depuis sa publication au Journal
officiel n° 7931 du 17 septembre 2026), à partir du texte officiel, en citant les
articles utilisés.
Projet construit étape par étape (RAG : *Retrieval-Augmented Generation*).

> Projet en cours de construction. Cet assistant est un outil d'information
> documentaire expérimental : ses réponses ne constituent pas un avis juridique.

## Feuille de route

- [x] Étape 1 — Découvrir les embeddings et la similarité cosinus
- [x] Étape 2 — Extraire les 462 articles du Journal officiel (PDF sur deux colonnes)
- [x] Étape 3 — Découper en passages et indexer dans PostgreSQL + pgvector
- [ ] Étape 4 — Générer des réponses sourcées avec un LLM (API FastAPI)
- [ ] Étape 5 — Évaluer sur ~20 questions (Recall@k, fidélité, abstention)
- [ ] Étape 6 — Garde-fous, function calling, observabilité

## Installation

Prérequis : Python 3.11+, Git, Docker Desktop (à partir de l'étape 3).

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Linux / macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

## Étape 1 — Découvrir les embeddings

```bash
python scripts/01_decouvrir_embeddings.py
```

Le script transforme une question et quelques passages en vecteurs, puis les
classe par similarité cosinus. On constate que le modèle compare le **sens** :
la question « jours de vacances » retrouve le passage sur le « congé payé ».

### Ce que l'étape 1 m'a appris

| Question posée | Passage « congé payé » | Rang |
|---|---|---|
| « Combien de jours de vacances un employé peut-il prendre ? » | 0,463 | 2e (derrière « 40 heures par semaine ») |
| « Le salarié a-t-il droit à des congés chaque année ? » | 0,783 | 1er |

Le modèle `paraphrase-multilingual-MiniLM-L12-v2` retrouve très bien un passage
quand la question lui ressemble, mais se trompe quand l'utilisateur emploie
d'autres mots (« vacances » au lieu de « congé payé »). Conséquences pour la suite :
récupérer plusieurs passages (top-k), choisir un modèle entraîné pour la recherche,
combiner recherche par le sens et par mots-clés, et évaluer avec des questions reformulées.

## Étape 2 — Extraire les articles du Code du travail

Source : [Journal officiel n° 7931 du 17 septembre 2026](https://www.vie-publique.sn/docs/56984d14-34a4-459e-8820-dbca07d9393c/JO-7931-du-17-septembre-2026.pdf)
(pages 1603 à 1674), placé dans `data/raw/`.

```bash
python src/ingest.py
```

Résultat : `data/processed/articles.jsonl`, un article par ligne, avec son numéro,
son intitulé, sa place dans le Code (Titre, Chapitre, Section, Sous-section,
Paragraphe), sa page dans le Journal officiel et une citation prête à l'emploi,
par exemple : *Code du travail 2026, article 277 (JO n° 7931, p. 1643)*.

Difficultés traitées :
- le Journal officiel est **sur deux colonnes** : le PDF est lu colonne par colonne ;
- en-têtes de page retirés, mots coupés en fin de ligne recollés ;
- intitulés écrits sur deux lignes, article dont le titre et le texte sont sur la même ligne ;
- une phrase comme « au titre de… » ne doit pas être confondue avec un intitulé « TITRE ».

Contrôles automatiques : 462 articles, aucun manquant, aucun doublon.

## Étape 3 — Découper en passages et indexer dans pgvector

```bash
cp .env.example .env          # puis compléter si besoin
docker compose up -d          # démarre PostgreSQL + pgvector (port 5433)
python src/chunking.py        # 462 articles -> 487 passages
python src/indexation.py      # embeddings + enregistrement dans la table "passages"
python src/recherche.py "Quelle est la durée du congé de maternité ?"
```

**Découpage (`src/chunking.py`)** : un article court reste un seul passage, pour ne pas
couper son sens juridique. Les 13 articles longs sont coupés entre deux paragraphes,
avec un chevauchement d'un paragraphe ; l'article 3 (Définitions) est coupé
définition par définition. Chaque passage reçoit un en-tête
(« Article 277 - Congé de maternité » + sa place dans le Code) avant d'être vectorisé.

**Embeddings** : `intfloat/multilingual-e5-small`, un modèle entraîné pour la recherche
question → passage (préfixes `query:` / `passage:`), choisi après les observations de
l'étape 1. Vecteurs de 384 dimensions, normalisés.

**Base vectorielle** : PostgreSQL 16 + pgvector, index HNSW en distance cosinus.
La recherche (`src/recherche.py`) renvoie les k passages les plus proches avec leur
score et leur citation.


### Ce que l'étape 3 m'a appris

| Question | Article 277 (Congé de maternité) |
|---|---|
| « Quelle est la durée du congé de maternité ? » | 2e (score 0,877) |
| « Combien de semaines de congé bébé pour une maman ? » | 2e (score 0,863) |

Avec `multilingual-e5-small`, entraîné pour la recherche, la question reformulée
retrouve le bon article, ce que le modèle de paraphrase de l'étape 1 ne faisait pas.
L'article 280 passe devant car il commence par « À la fin du congé de maternité » :
la recherche ramène des candidats, c'est le LLM (étape 4) qui choisira le bon passage.
Les scores de e5 sont très serrés (0,84 à 0,89) : le seuil de refus des questions
hors sujet devra être calibré, il ne peut pas être fixé à l'intuition.
