# Résultats de l'évaluation

Modèle : `openai/gpt-oss-120b` — 20 questions (18 avec réponse dans le Code, 2 hors corpus) — réécriture de la question : **oui**.

## Recherche (retrieval)
- Recall@5 : **18/18 (100 %)**
- Recall@5 strict (tous les articles attendus) : 17/18 (94 %)

## Réponses
- Citation d'un article attendu : **18/18 (100 %)**
- Réponses contenant une citation inexistante : 2/20
- Abstention correcte sur les questions hors corpus : **2/2 (100 %)**
- Abstentions inutiles (refus alors que la réponse existe) : 0/18
- Notes du LLM juge (sur 2) : exactitude **1.94**, complétude **1.78**, fidélité **2.00**
- Durée moyenne du LLM : 8.07 s — tokens moyens par question : 2220.95

## Par type de question

| Type | n | Recall@5 | Citation correcte | Exactitude moy. |
|---|---|---|---|---|
| factuelle | 5 | 5/5 (100 %) | 5/5 (100 %) | 2.00 |
| procedure | 4 | 4/4 (100 %) | 4/4 (100 %) | 2.00 |
| conditions | 4 | 4/4 (100 %) | 4/4 (100 %) | 2.00 |
| multi_passages | 3 | 3/3 (100 %) | 3/3 (100 %) | 2.00 |
| absente | 2 | — | — | — |
| reformulee | 2 | 2/2 (100 %) | 2/2 (100 %) | 1.50 |

## Questions à analyser

- **R01** (reformulee) : attendus [241, 244], récupérés [251, 242, 243, 298, 246, 244], cités [242, 244, 246, 298], exactitude 1 — Réponse correcte sur le principe mais introduit une exception non mentionnée dans la référence, ce qui constitue une petite inexactitude.

Limites : 20 questions seulement ; notes produites par un LLM juge (le même modèle que l'assistant), à vérifier à la main sur un échantillon (colonne `verif_humaine` du CSV).
