"""
Démo interactive du pipeline RAG.

Permet de tester manuellement le système avant de lancer la suite de tests :
    python demo.py
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

from app.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("  RAG-TestKit — Démo interactive")
    print("=" * 60)
    print("\n⏳ Chargement des modèles (2-5 min au premier lancement)...\n")

    pipeline = RAGPipeline()
    pipeline.initialize()

    print("\n✅ Pipeline prêt ! Posez vos questions (ou 'exit' pour quitter)\n")
    print("Exemples :")
    print("  - Combien de jours de congés par an ?")
    print("  - Quelle est la politique de télétravail ?")
    print("  - Quelle est la capitale de l'Australie ?  (doit être refusée)\n")

    while True:
        question = input("❓ Question > ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Au revoir !")
            break
        if not question:
            continue

        result = pipeline.query(question)

        print(f"\n💬 Réponse ({result['latency_seconds']}s) :")
        print(f"   {result['answer']}\n")
        print(f"📄 Sources utilisées : {', '.join(result['sources'])}")
        print(f"📊 Scores de pertinence : {[round(s, 3) for s in result['retrieval_scores']]}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
