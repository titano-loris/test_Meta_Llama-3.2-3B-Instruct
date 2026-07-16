"""
tests RAPIDES (pas de LLM), afin de valider
la couche de recherche documentaire.

Lancement : pytest tests/unit/ -v -m unit
"""
import pytest

from app.config import TOP_K_DOCUMENTS


@pytest.mark.unit
class TestRetrieverBasics:
    """Comportement nominal du retriever."""

    def test_retriever_returns_documents(self, retriever):
        """Une question valide doit retourner au moins 1 document."""
        results = retriever.retrieve("Combien de jours de congés par an ?")
        assert len(results) > 0, "Le retriever n'a retourné aucun document"

    def test_retriever_respects_top_k(self, retriever):
        """Le nombre de documents retournés ne doit jamais dépasser top_k en gros 3."""
        results = retriever.retrieve("politique de l'entreprise")
        assert len(results) <= TOP_K_DOCUMENTS

    def test_retriever_result_structure(self, retriever):
        """Chaque résultat doit contenir : content, source et score."""
        results = retriever.retrieve("télétravail")
        for doc in results:
            assert "content" in doc
            assert "source" in doc
            assert "score" in doc
            assert isinstance(doc["score"], float)

    def test_scores_are_sorted_descending(self, retriever):
        """Le meilleur document est en premier, Le plus pertinent est en haut de la pile"""
        results = retriever.retrieve("formation budget annuel")
        scores = [doc["score"] for doc in results]
        assert scores == sorted(scores, reverse=True), (
            "Les scores ne sont pas triés par ordre décroissant"
        )


@pytest.mark.unit
class TestRetrieverRelevance:
    """Pertinence sémantique : la bonne source doit remonter en premier."""

    @pytest.mark.parametrize(
        "question,expected_source",
        [
            ("Combien de jours de congés payés ?", "politique_conges.md"),
            ("Combien de jours de télétravail par semaine sont autorisés ?", "politique_teletravail.md"),
            ("Comment se passe l'intégration d'un nouveau ?", "processus_onboarding.md"),
            ("Quel est le budget formation annuel par salarié ?", "politique_formation.md"),
            ("Comment me faire rembourser un restaurant client ?", "remboursement_frais.md"),
        ],
    )
    def test_top_document_matches_expected_source(
        self, retriever, question, expected_source
    ):
        """Le document le plus pertinent doit provenir de la bonne source."""
        results = retriever.retrieve(question)
        assert results, f"Aucun résultat pour : {question}"
        assert results[0]["source"] == expected_source, (
            f"Question : '{question}'\n"
            f"Source attendue : {expected_source}\n"
            f"Source obtenue : {results[0]['source']}"
        )


@pytest.mark.unit
class TestRetrieverEdgeCases:
    """Robustesse du retriever face aux inputs limites."""

    def test_empty_query_returns_empty_list(self, retriever):
        """Une requête vide ne doit pas planter — retour liste vide."""
        assert retriever.retrieve("") == []

    def test_whitespace_query_returns_empty_list(self, retriever):
        """Une requête composée d'espaces doit être traitée comme vide."""
        assert retriever.retrieve("   ") == []

    def test_very_long_query_does_not_crash(self, retriever):
        """Une requête très longue ne doit pas faire planter l'index."""
        long_query = "congés payés " * 200
        results = retriever.retrieve(long_query)
        assert isinstance(results, list)

    def test_special_characters_do_not_crash(self, retriever):
        """Caractères spéciaux et payloads ne doivent pas planter."""
        for query in ["<script>alert(1)</script>", "'; DROP TABLE--", "🎉🎊"]:
            results = retriever.retrieve(query)
            assert isinstance(results, list)
