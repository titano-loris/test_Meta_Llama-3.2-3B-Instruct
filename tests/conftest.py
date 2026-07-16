"""
Fixtures pytest partagées par toute la suite de tests.
Point clé d'architecture : les fixtures lourdes (retriever, pipeline)
sont en scope "session" — les modèles ne sont chargés qu'UNE fois
pour toute la session de test:
Sans ça, chaque test chargerait le modèle MiniLM indépendamment,13 chargements au lieu de 1.
"""
import json
import sys
from pathlib import Path

import pytest

# Rend le package `app` importable depuis les tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    ADVERSARIAL_INPUTS_PATH,
    GOLDEN_DATASET_PATH,
)


# ---------- Datasets (légers, scope function) ----------

@pytest.fixture
def golden_dataset() -> list[dict]:
    """Jeu de données de référence : questions + réponses attendues."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def adversarial_inputs() -> dict:
    """Inputs malveillants et cas limites pour les tests de sécurité."""
    with open(ADVERSARIAL_INPUTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------- Composants lourds (scope session) ----------

@pytest.fixture(scope="session")
def retriever():
    """
    Retriever initialisé une seule fois pour toute la session.
    Chargement : ~10 secondes (embeddings MiniLM).
    """
    from app.retriever import Retriever

    r = Retriever()
    r.load_knowledge_base()
    return r


@pytest.fixture(scope="session")
def rag_pipeline():
    """
    Pipeline RAG complet initialisé une seule fois.
    ATTENTION : charge le LLM en mémoire (~8GB RAM, plusieurs minutes).
    Les tests marqués @pytest.mark.slow utilisent cette fixture.
    """
    from app.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline
