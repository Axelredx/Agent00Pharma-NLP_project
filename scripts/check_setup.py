import os, glob
import requests
from dotenv import load_dotenv

load_dotenv()
ok = True

def check(name, fn):
    global ok
    try:
        print(f"[OK]   {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

def as_text(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

def gemini(env_var):
    def run():
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=os.environ[env_var], temperature=0)
        return as_text(llm.invoke("Rispondi solo con la parola: pronto").content)
    return run

def rxnorm():
    r = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json",
                     params={"name": "warfarin"}, timeout=20)
    r.raise_for_status()
    return r.json()["idGroup"].get("rxnormId")

def dailymed():
    r = requests.get("https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
                     params={"drug_name": "warfarin", "pagesize": 1}, timeout=20)
    r.raise_for_status()
    return r.json()["data"][0]["title"]

def dataset():
    n_train = len(glob.glob("data/raw/DDICorpus/Train/**/*.xml", recursive=True))
    n_test = len(glob.glob("data/raw/DDICorpus/Test/Test for DDI Extraction task/**/*.xml", recursive=True))
    assert n_train > 700 and n_test > 150, "dataset non trovato o incompleto"
    return f"{n_train} file train, {n_test} file test"

check("Gemini (estrattore)", gemini("GEMINI_EXTRACTOR_MODEL"))
check("Gemini (judge)", gemini("GEMINI_JUDGE_MODEL"))
check("RxNorm", rxnorm)
check("DailyMed", dailymed)
check("Dataset DDI", dataset)
print("\nTutto a posto." if ok else "\nQualcosa da sistemare (vedi sopra).")
