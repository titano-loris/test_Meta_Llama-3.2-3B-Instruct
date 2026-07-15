"""
Tests d'évaluation — Hallucination & Fidélité au contexte.

Approche à deux niveaux :
1. Tests DÉTERMINISTES (sans juge LLM) : vérification de la présence
   des faits attendus (must_contain) et du refus sur questions hors périmètre.
2. Tests avec MÉTRIQUES DeepEval (optionnels, nécessitent un modèle juge).

Le niveau 1 suffit pour un CI reproductible. Le niveau 2 apporte
des scores plus fins mais introduit une dépendance à un LLM externe.
"""
import pytest


@pytest.mark.evaluation
@pytest.mark.slow
class TestFactualAccuracy:
    """Niveau 1 — Vérifications déterministes sur le golden dataset."""

    def test_answers_contain_expected_facts(self, rag_pipeline, golden_dataset):
        """
        Pour chaque question factuelle du golden dataset, la réponse
        doit contenir au moins un des éléments 'must_contain'.
        """
        failures = []

        for sample in golden_dataset:
            if sample["category"] == "hors_perimetre":
                continue  # traité par le test suivant

            result = rag_pipeline.query(sample["question"])
            answer_lower = result["answer"].lower()

            found = any(
                token.lower() in answer_lower for token in sample["must_contain"]
            )
            if not found:
                failures.append(
                    f"[{sample['id']}] '{sample['question']}'\n"
                    f"  Attendu (un parmi) : {sample['must_contain']}\n"
                    f"  Obtenu : {result['answer'][:200]}"
                )

        assert not failures, (
            f"{len(failures)} réponse(s) factuellement incorrecte(s) :\n\n"
            + "\n\n".join(failures)
        )


@pytest.mark.evaluation
@pytest.mark.slow
class TestHallucinationResistance:
    """Le modèle doit REFUSER de répondre hors de sa base de connaissances."""

    def test_out_of_scope_question_is_refused(self, rag_pipeline, golden_dataset):
        """
        Question hors périmètre (Q010 : couleur du logo) —
        le modèle doit répondre par le refus standard, pas inventer.
        """
        out_of_scope = [
            s for s in golden_dataset if s["category"] == "hors_perimetre"
        ]
        assert out_of_scope, "Le golden dataset doit contenir des questions hors périmètre"

        for sample in out_of_scope:
            result = rag_pipeline.query(sample["question"])
            answer_lower = result["answer"].lower()

            refused = any(
                token.lower() in answer_lower for token in sample["must_contain"]
            )
            assert refused, (
                f"HALLUCINATION DÉTECTÉE — le modèle a répondu à une question "
                f"hors périmètre au lieu de refuser.\n"
                f"Question : {sample['question']}\n"
                f"Réponse : {result['answer'][:300]}"
            )

    def test_general_knowledge_questions_are_refused(
        self, rag_pipeline, adversarial_inputs
    ):
        """
        Les questions de culture générale (capitale, poème...) doivent
        être refusées — le modèle ne doit pas puiser dans ses connaissances
        d'entraînement.
        """
        refusal_markers = ["ne dispose pas", "pas cette information", "base de connaissances"]

        for case in adversarial_inputs["out_of_scope"]:
            result = rag_pipeline.query(case["input"])
            answer_lower = result["answer"].lower()

            refused = any(marker in answer_lower for marker in refusal_markers)
            assert refused, (
                f"[{case['id']}] Le modèle a répondu hors de sa base :\n"
                f"Question : {case['input']}\n"
                f"Réponse : {result['answer'][:300]}"
            )


# ---------------------------------------------------------------------------
# Niveau 2 — Métriques DeepEval (optionnel)
# Décommenter si un modèle juge est configuré (OpenAI ou LLM local via Ollama).
# ---------------------------------------------------------------------------
#
# from deepeval.metrics import FaithfulnessMetric
# from deepeval.test_case import LLMTestCase
# from app.config import FAITHFULNESS_THRESHOLD
#
# @pytest.mark.evaluation
# @pytest.mark.slow
# @pytest.mark.requires_judge
# def test_faithfulness_with_deepeval(rag_pipeline, golden_dataset):
#     """Score de fidélité mesuré par un LLM juge (DeepEval)."""
#     sample = golden_dataset[0]
#     result = rag_pipeline.query(sample["question"])
#
#     test_case = LLMTestCase(
#         input=sample["question"],
#         actual_output=result["answer"],
#         retrieval_context=result["context_docs"],
#     )
#     metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD)
#     metric.measure(test_case)
#     assert metric.score >= FAITHFULNESS_THRESHOLD
