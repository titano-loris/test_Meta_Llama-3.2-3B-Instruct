"""
Configuration centrale du projet RAG-TestKit.
Toutes les constantes et paramètres sont définis ici.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Chemins ---
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
KNOWLEDGE_BASE_PATH = DATASETS_DIR / "knowledge_base.json"
GOLDEN_DATASET_PATH = DATASETS_DIR / "golden_dataset.json"
ADVERSARIAL_INPUTS_PATH = DATASETS_DIR / "adversarial_inputs.json"

# --- Modèles ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Paramètres RAG ---
TOP_K_DOCUMENTS = 3          # Nombre de documents récupérés par requête
MAX_NEW_TOKENS = 256         # Longueur max de la réponse générée
TEMPERATURE = 0.1            # Basse température = réponses déterministes (important pour les tests)

# --- Seuils de qualité (utilisés par les tests) ---
FAITHFULNESS_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 0.3
RELEVANCE_THRESHOLD = 0.6
