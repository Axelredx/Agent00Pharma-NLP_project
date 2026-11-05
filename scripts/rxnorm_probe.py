"""Prova dal vivo del client RxNorm su alcuni nomi (serve connessione internet).

    python -m scripts.rxnorm_probe

Serve a controllare che il parsing delle risposte dell'API sia corretto prima di
lanciare lo studio completo. Stampa, per ogni nome, i codici e i principi attivi.
"""
from src.tools.rxnorm import RxNormClient, RxNormError, same_drug

NAMES = ["warfarin", "Coumadin", "warfarin sodium", "aspirin", "acetylsalicylic acid",
         "Tylenol", "acetaminophen", "paracetamol", "INH", "isoniazid",
         "Bactrim", "trimethoprim", "quinolones", "magnesium"]

PAIRS = [("Coumadin", "warfarin"), ("warfarin sodium", "warfarin"), ("aspirin", "acetylsalicylic acid"),
         ("Tylenol", "acetaminophen"), ("acetaminophen", "paracetamol"), ("INH", "isoniazid"),
         ("Bactrim", "trimethoprim"), ("warfarin", "aspirin")]


def main() -> None:
    client = RxNormClient()
    try:
        print(f"{'nome':<22}{'stato':<15}{'rxcui':<20}principi attivi")
        for n in NAMES:
            r = client.resolve(n)
            ings = ", ".join(f"{name} ({c})" for c, name in r.ingredient_names.items()) or "-"
            print(f"{n:<22}{r.status:<15}{','.join(r.rxcuis)[:18]:<20}{ings}")
        print("\nstesso farmaco?")
        for a, b in PAIRS:
            ok, why = same_drug(a, b, client)
            print(f"  {a!r:<26} vs {b!r:<26} -> {ok!s:<6} ({why})")
    except RxNormError as e:
        print(f"\nERRORE: {e}")
        raise SystemExit(1)
    print(f"\nChiamate HTTP: {client.http_calls}, dalla cache: {client.cache_hits}")


if __name__ == "__main__":
    main()
