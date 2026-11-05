"""Configurazione centrale del progetto: percorsi, costanti, variabili d'ambiente."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Percorsi ---
RAW_DIR = ROOT / "data" / "raw" / "DDICorpus"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

TRAIN_XML_DIR = RAW_DIR / "Train"
TEST_XML_DIR = RAW_DIR / "Test" / "Test for DDI Extraction task"

# --- Etichette del task (DDI Extraction 2013) ---
NEGATIVE_LABEL = "none"
POSITIVE_LABELS = ["effect", "mechanism", "advise", "int"]
ALL_LABELS = [NEGATIVE_LABEL] + POSITIVE_LABELS

# --- Parametri sperimentali ---
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
GEMINI_EXTRACTOR_MODEL = os.getenv("GEMINI_EXTRACTOR_MODEL", "")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "")
# Richieste al minuto che il programma si impone (free tier: guarda i limiti reali
# nella dashboard di AI Studio e adatta il valore nel .env).
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "6"))
# Richieste al minuto per il modello del judge (controlla il limite nella dashboard di AI Studio).
GEMINI_JUDGE_RPM = float(os.getenv("GEMINI_JUDGE_RPM", "5"))
