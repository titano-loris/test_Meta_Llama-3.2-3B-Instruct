"""
Tests d'intégration du pipeline RAG complet.

Ces tests chargent le LLM — ils sont LENTS (30-90s par requête sur CPU).
Marqués @pytest.mark.slow pour pouvoir les exclure en développement :

    pytest -m "not slow"          → tests rapides uniquement
    pytest tests/integration/ -v  → tests d'intégration complets
"""
import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineEndToEnd:
    """Le pipeline complet : question → réponse structurée."""

    def test_pipeline_returns_complete_structure(self, rag_pipeline):
        """La réponse doit contenir tous les champs attendus."""
        result = rag_pipeline.query("Combien de jours de congés par an ?")

        expected_keys = {
            "question",
            "answer",
            "context_docs",
            "sources",
            "retrieval_scores",
            "latency_seconds",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Champs manquants : {expected_keys - result.keys()}"
        )

    def test_answer_is_non_empty_string(self, rag_pipeline):
        """La réponse générée ne doit jamais être vide."""
        result = rag_pipeline.query("Quelle est la politique de télétravail ?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"].strip()) > 0

    def test_context_docs_are_provided(self, rag_pipeline):
        """Le pipeline doit tracer les documents utilisés (auditabilité)."""
        result = rag_pipeline.query("Quel est le budget formation ?")
        assert len(result["context_docs"]) > 0
        assert len(result["sources"]) == len(result["context_docs"])

    def test_factual_answer_contains_expected_value(self, rag_pipeline):
        """Test factuel simple : la réponse doit contenir le chiffre clé."""
        result = rag_pipeline.query("Combien de jours de congés payés par an ?")
        assert "25" in result["answer"], (
            f"La réponse ne contient pas '25' : {result['answer']}"
        )

    def test_latency_is_measured(self, rag_pipeline):
        """La latence doit être mesurée et positive."""
        result = rag_pipeline.query("Quand ont lieu les entretiens annuels ?")
        assert result["latency_seconds"] > 0
