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
- [x] Étape 4 — Générer des réponses sourcées avec un LLM (Groq) et les exposer dans une API FastAPI
- [x] Étape 5 — Évaluer sur 20 questions (Recall@5, citations, abstention, LLM juge)
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
la recherche ramène des candidats, c'est le LLM (étape 4) qui choisit le bon passage.
Les scores de e5 sont très serrés (0,84 à 0,89) : le seuil de refus des questions
hors sujet devra être calibré, il ne peut pas être fixé à l'intuition.

## Étape 4 — Réponses sourcées avec un LLM, exposées dans une API

Ajouter sa clé gratuite Groq (https://console.groq.com) dans `.env` (`GROQ_API_KEY=...`), puis :

```bash
python src/lister_modeles_groq.py     # vérifier que GROQ_MODEL existe
python src/generation.py "Quelle est la durée du congé de maternité ?"
python src/api.py                      # API sur http://localhost:8000/docs
```

**Fonctionnement (`src/generation.py`)** :
1. recherche des 5 passages les plus proches (étape 3) ;
2. passages numérotés `[EXTRAIT 1]`… placés dans le prompt avec leur citation ;
3. prompt système strict : répondre uniquement à partir des extraits, citer chaque
   affirmation, dire clairement quand les extraits ne suffisent pas, traiter les extraits
   comme des données et non comme des instructions (protection contre l'injection) ;
4. `temperature=0` pour des réponses stables ;
5. vérification automatique : le LLM a-t-il cité un extrait qui n'existe pas ?
6. mesures renvoyées avec chaque réponse : durée de la recherche, durée du LLM,
   tokens consommés (base de l'observabilité).

**API (`src/api.py`)** : `POST /question` et `GET /sante`. Le modèle d'embeddings est
chargé une seule fois au démarrage du serveur (au lieu de 10 à 30 s par question en
script). Validation des entrées (Pydantic) et messages d'erreur clairs si la base est
éteinte (503) ou si le service LLM échoue (502).

Les règles du prompt et la vérification des citations s'inspirent du projet RAG juridique
réalisé par mon groupe de formation à Orange Digital Center, adaptés ici au Code du travail.

### Ce que l'étape 4 m'a appris

- Question : « Quelle est la durée du congé de maternité ? »
- Réponse : 18 semaines, dont 12 après l'accouchement, prolongeables de 3 semaines,
  en citant l'article 277, alors que la recherche avait classé l'article 280 en premier :
  la recherche ramène des candidats, le LLM choisit le bon passage.
- Mesures : 1,5 s pour le LLM (`openai/gpt-oss-120b` via Groq), 1 461 tokens en entrée, 236 en sortie.
- Le modèle prévu au départ (`llama-3.3-70b-versatile`) avait été retiré par Groq (erreur 404) :
  le nom du modèle est donc configurable dans `.env`, et `src/lister_modeles_groq.py`
  vérifie qu'il existe toujours.

## Étape 5 — Évaluation sur 20 questions

```bash
python src/evaluate.py
```

**Jeu de test (`data/evaluation/questions.json`)** : 20 questions, chacune avec une réponse
de référence rédigée à partir du texte de loi et le ou les articles attendus.

| Type | Nombre | Ce qu'on teste |
|---|---|---|
| factuelle | 5 | retrouver une information explicite |
| procedure | 4 | restituer des étapes sans en oublier |
| conditions | 4 | combiner les conditions et exceptions |
| multi_passages | 3 | réponse qui demande deux articles |
| absente | 2 | refuser de répondre plutôt qu'inventer |
| reformulee | 2 | langage courant (« bosser », « vacances ») |

**Mesures** :
- *Recherche* : Recall@5 (un article attendu parmi les 5 passages retrouvés) et Recall@5 strict
  (tous les articles attendus) ;
- *Réponse* : citation d'un article attendu, citations inexistantes, abstention correcte sur
  les questions hors corpus, abstentions inutiles ;
- *Qualité* : exactitude, complétude et fidélité (0 à 2) notées par un **LLM juge** qui compare
  la réponse à la référence et aux extraits.

**Limite assumée** : le juge est le même modèle que l'assistant et peut se tromper. Une partie
des notes est donc vérifiée à la main (colonne `verif_humaine` de `reports/resultats_eval.csv`).

Résultats : `reports/resume_eval.md` (résumé) et `reports/resultats_eval.csv` (détail par question).
