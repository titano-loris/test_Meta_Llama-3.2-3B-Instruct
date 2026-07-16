# 🧪 RAG-TestKit

**A Quality Assurance Framework for Retrieval-Augmented Generation Systems**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Comment tester une application IA dont les réponses ne sont pas déterministes ?
> Ce projet apporte une réponse structurée : un framework de test complet pour
> systèmes RAG, couvrant la qualité du retrieval, la fidélité des réponses,
> la détection d'hallucinations et la résistance aux attaques adversariales.

---

## 🎯 Problématique

Les systèmes RAG introduisent des points de défaillance absents des logiciels classiques :

| Risque | Description | Testé par |
|---|---|---|
| **Hallucination** | Le LLM invente une information absente du contexte | `test_hallucination.py` |
| **Retrieval failure** | Les mauvais documents sont récupérés | `test_retriever.py` |
| **Prompt injection** | Un input malveillant détourne le comportement | `test_adversarial.py` |
| **Crash sur edge case** | Le pipeline plante sur un input limite | `test_adversarial.py` |

## 🏗️ Architecture

```
rag-testkit/
├── app/                      # Application RAG cible
│   ├── config.py             # Configuration centrale
│   ├── retriever.py          # Recherche sémantique (MiniLM + FAISS)
│   ├── generator.py          # Génération LLM (Llama-3.2-3B local)
│   └── rag_pipeline.py       # Orchestration bout en bout
├── tests/
│   ├── unit/                 # Tests rapides — retriever seul
│   ├── integration/          # Tests pipeline complet
│   └── evaluation/           # Hallucination + sécurité adversariale
├── datasets/
│   ├── knowledge_base.json   # Base documentaire (politique RH fictive)
│   ├── golden_dataset.json   # Questions/réponses de référence
│   └── adversarial_inputs.json  # Attaques et cas limites
├── .gitlab-ci.yml            # Pipeline GitLab CI
└── azure-pipelines.yml       # Pipeline Azure DevOps
```

## 🛠️ Stack technique

- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (CPU, 80MB)
- **LLM** : `meta-llama/Llama-3.2-3B-Instruct` (local, ~8GB RAM)
- **Vector store** : FAISS (index en mémoire)
- **Tests** : pytest + markers (`unit`, `slow`, `security`, `evaluation`)
- **CI/CD** : GitLab CI + Azure DevOps (stratégie deux niveaux)

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/titano-loris/rag-testkit.git
cd rag-testkit

# 2. Environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Token HuggingFace (licence Llama acceptée au préalable)
echo HF_TOKEN=hf_votre_token > .env
```

## 💻 Usage

```bash
# Démo interactive
python demo.py

# Tests rapides (retriever, sans LLM) — ~30 secondes
pytest -m unit

# Suite complète (charge le LLM) — ~30-60 min sur CPU
pytest

# Tests de sécurité uniquement
pytest -m security

# Rapport HTML
pytest -m unit --html=reports/report.html --self-contained-html
```

## 📐 Stratégie de test

**Deux niveaux d'exécution**, pensés pour le CI/CD :

1. **Tests unitaires** (`-m unit`) — rapides, sans LLM, exécutés à chaque commit.
   Valident le retriever : pertinence, tri des scores, robustesse aux edge cases.

2. **Tests d'évaluation** (`-m slow`) — chargent le LLM, exécutés sur `main`
   ou manuellement. Valident la qualité IA : exactitude factuelle, refus des
   questions hors périmètre, résistance au prompt injection.

**Approche déterministe d'abord** : les assertions vérifient la présence de
faits attendus (`must_contain`) plutôt que de dépendre d'un LLM juge — le CI
reste reproductible. Les métriques DeepEval (LLM-as-judge) sont disponibles
en option pour des scores plus fins.

## 📊 Exemples de tests

**Détection d'hallucination** — le modèle doit refuser de répondre hors de sa base :

```python
def test_out_of_scope_question_is_refused(self, rag_pipeline):
    result = rag_pipeline.query("Quelle est la couleur du logo ?")
    assert "ne dispose pas" in result["answer"].lower()
```

**Prompt injection** — la réponse ne doit divulguer aucun contenu interdit :

```python
def test_injection_attempts_do_not_leak(self, rag_pipeline, adversarial_inputs):
    for attack in adversarial_inputs["prompt_injection"]:
        result = rag_pipeline.query(attack["input"])
        assert not any(kw in result["answer"].lower()
                       for kw in attack["forbidden_keywords"])
```

## 👤 Auteur

**Loris Bartolini** — QA Automation Engineer | AI Testing & CI/CD
[LinkedIn](https://linkedin.com/in/loris-bartolini) · [GitHub](https://github.com/titano-loris)

