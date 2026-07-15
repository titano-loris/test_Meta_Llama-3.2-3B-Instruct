"""
RAG Pipeline — Orchestration retriever + generator.

C'est le composant testé de bout en bout par la suite de tests.
Il expose une interface simple : query(question) → réponse structurée.
"""
import logging
import time

from app.generator import Generator
from app.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG complet : question → retrieval → génération → réponse.

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()
        result = pipeline.query("Quelle est la politique de télétravail ?")
        print(result["answer"])
    """

    def __init__(self):
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Charge les modèles et construit l'index. À appeler une seule fois."""
        logger.info("Initialisation du pipeline RAG...")
        self.retriever = Retriever()
        self.retriever.load_knowledge_base()
        self.generator = Generator()
        self._initialized = True
        logger.info("Pipeline RAG prêt.")

    def query(self, question: str) -> dict:
        """
        Traite une question de bout en bout.

        Returns:
            {
                "question": str,
                "answer": str,
                "context_docs": list[str],   # contenus des documents utilisés
                "sources": list[str],        # noms des documents sources
                "retrieval_scores": list[float],
                "latency_seconds": float,
            }
        """
        if not self._initialized:
            raise RuntimeError("Pipeline non initialisé. Appeler initialize() d'abord.")

        start = time.perf_counter()

        # Étape 1 — Retrieval
        retrieved = self.retriever.retrieve(question)
        context_docs = [doc["content"] for doc in retrieved]
        sources = [doc["source"] for doc in retrieved]
        scores = [doc["score"] for doc in retrieved]

        # Étape 2 — Génération
        answer = self.generator.generate(question, context_docs)

        latency = time.perf_counter() - start

        return {
            "question": question,
            "answer": answer,
            "context_docs": context_docs,
            "sources": sources,
            "retrieval_scores": scores,
            "latency_seconds": round(latency, 2),
        }
