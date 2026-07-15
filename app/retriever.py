"""
Retriever — Composant de recherche documentaire.

Responsabilité : encoder les documents de la base de connaissances
en vecteurs (embeddings), puis retrouver les documents les plus
pertinents pour une question donnée via une recherche de similarité.

Stack : sentence-transformers (embeddings) + FAISS (index vectoriel)
"""
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, KNOWLEDGE_BASE_PATH, TOP_K_DOCUMENTS

logger = logging.getLogger(__name__)


class Retriever:
    """
    Moteur de recherche sémantique basé sur FAISS.

    Usage:
        retriever = Retriever()
        retriever.load_knowledge_base()
        docs = retriever.retrieve("Quelle est la politique de congés ?")
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        logger.info(f"Chargement du modèle d'embeddings : {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []

    def load_knowledge_base(self, path: Path = KNOWLEDGE_BASE_PATH) -> None:
        """Charge les documents et construit l'index FAISS."""
        with open(path, encoding="utf-8") as f:
            self.documents = json.load(f)

        if not self.documents:
            raise ValueError("La base de connaissances est vide.")

        texts = [doc["content"] for doc in self.documents]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,  # normalisation → produit scalaire = similarité cosinus
            show_progress_bar=False,
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))

        logger.info(f"Index FAISS construit : {len(self.documents)} documents, dim={dimension}")

    def retrieve(self, query: str, top_k: int = TOP_K_DOCUMENTS) -> list[dict]:
        """
        Retourne les top_k documents les plus pertinents pour la requête.

        Returns:
            Liste de dicts : [{"content": str, "source": str, "score": float}, ...]
        """
        if self.index is None:
            raise RuntimeError("Index non initialisé. Appeler load_knowledge_base() d'abord.")

        if not query or not query.strip():
            logger.warning("Requête vide reçue par le retriever.")
            return []

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        scores, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS retourne -1 si moins de documents que top_k
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "content": doc["content"],
                    "source": doc.get("source", "unknown"),
                    "score": float(score),
                }
            )

        return results
