# Résultats de l'évaluation

Modèle : `openai/gpt-oss-120b` — 20 questions (18 avec réponse dans le Code, 2 hors corpus).

## Recherche (retrieval)
- Recall@5 : **17/18 (94 %)**
- Recall@5 strict (tous les articles attendus) : 17/18 (94 %)

## Réponses
- Citation d'un article attendu : **17/18 (94 %)**
- Réponses contenant une citation inexistante : 0/20
- Abstention correcte sur les questions hors corpus : **2/2 (100 %)**
- Abstentions inutiles (refus alors que la réponse existe) : 1/18
- Notes du LLM juge (sur 2) : exactitude **1.78**, complétude **1.67**, fidélité **1.94**
- Durée moyenne du LLM : 6.95 s — tokens moyens par question : 1482.75

## Par type de question

| Type | n | Recall@5 | Citation correcte | Exactitude moy. |
|---|---|---|---|---|
| factuelle | 5 | 5/5 (100 %) | 5/5 (100 %) | 1.80 |
| procedure | 4 | 4/4 (100 %) | 4/4 (100 %) | 1.75 |
| conditions | 4 | 4/4 (100 %) | 4/4 (100 %) | 2.00 |
| multi_passages | 3 | 3/3 (100 %) | 3/3 (100 %) | 2.00 |
| absente | 2 | — | — | — |
| reformulee | 2 | 1/2 (50 %) | 1/2 (50 %) | 1.00 |

## Questions à analyser

- **F01** (factuelle) : attendus [277], récupérés [280, 277, 274, 249, 253], cités [277], exactitude 1 — Réponse correcte mais omet la mention "ou aux couches" dans les cas de prolongation.
- **P02** (procedure) : attendus [127, 129], récupérés [127, 129, 128, 131, 110], cités [127, 128, 129, 131], exactitude 1 — L’assistant ajoute une étape sur le plan social qui n’est pas dans la réponse de référence, ce qui constitue une légère imprécision, mais le reste est complet et fidèle aux extraits.
- **R01** (reformulee) : attendus [241, 244], récupérés [251, 242, 243, 298, 246], cités [242, 246, 298], exactitude 0 — L’assistant ne répond pas à la question centrale (la durée légale de 40 h et l’obligation de majoration), omet l’essentiel et conclut à tort qu’on ne peut pas déterminer la réponse à partir des extraits, alors que la référence indique clairement la règle.

Limites : 20 questions seulement ; notes produites par un LLM juge (le même modèle que l'assistant), à vérifier à la main sur un échantillon (colonne `verif_humaine` du CSV).
