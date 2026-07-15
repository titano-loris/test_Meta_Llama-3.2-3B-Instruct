"""
Tests adversariaux — Sécurité & Robustesse.

Trois familles d'attaques testées :
1. Prompt injection — tentatives de détournement du comportement
2. Edge cases — inputs limites (vide, très long, payloads)
3. Exfiltration — tentatives de révélation du prompt système

Lancement : pytest tests/evaluation/test_adversarial.py -v -m security
"""
import pytest


@pytest.mark.security
@pytest.mark.slow
class TestPromptInjection:
    """Résistance aux tentatives de détournement du modèle."""

    def test_injection_attempts_do_not_leak_forbidden_content(
        self, rag_pipeline, adversarial_inputs
    ):
        """
        Chaque attaque du dataset adversarial est jouée contre le pipeline.
        La réponse ne doit contenir AUCUN des mots-clés interdits associés.
        """
        breaches = []

        for attack in adversarial_inputs["prompt_injection"]:
            result = rag_pipeline.query(attack["input"])
            answer_lower = result["answer"].lower()

            leaked = [
                kw for kw in attack["forbidden_keywords"]
                if kw.lower() in answer_lower
            ]
            if leaked:
                breaches.append(
                    f"[{attack['id']}] {attack['description']}\n"
                    f"  Attaque : {attack['input'][:100]}\n"
                    f"  Fuites détectées : {leaked}\n"
                    f"  Réponse : {result['answer'][:200]}"
                )

        assert not breaches, (
            f"{len(breaches)} FAILLE(S) DE SÉCURITÉ détectée(s) :\n\n"
            + "\n\n".join(breaches)
        )


@pytest.mark.security
@pytest.mark.slow
class TestEdgeCaseRobustness:
    """Le pipeline ne doit jamais planter, quel que soit l'input."""

    def test_edge_cases_do_not_crash_pipeline(
        self, rag_pipeline, adversarial_inputs
    ):
        """
        Inputs limites : vide, espaces, répétitions, SQL, XSS, emojis.
        Critère : le pipeline retourne toujours une réponse structurée.
        """
        for case in adversarial_inputs["edge_cases"]:
            try:
                result = rag_pipeline.query(case["input"])
                assert result is not None, f"[{case['id']}] Résultat None"
                assert "answer" in result, f"[{case['id']}] Champ 'answer' absent"
                assert isinstance(result["answer"], str), (
                    f"[{case['id']}] La réponse n'est pas une chaîne"
                )
            except Exception as exc:
                pytest.fail(
                    f"[{case['id']}] CRASH sur '{case['description']}' : {exc}"
                )
