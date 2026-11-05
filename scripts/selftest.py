"""Test offline della logica (nessuna chiamata API, nessuna quota consumata).

    python -m scripts.selftest
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import eval.run_eval as run_eval
from config import PROCESSED_DIR
from src.llm import CachedLLM, DailyQuotaExceeded
from src.prompts import build_baseline_prompt, mark_entities, select_fewshot
from src.schemas import INVALID_LABEL, parse_prediction


def ok(msg):
    print(f"[OK] {msg}")


ok_ = ok


def test_marking():
    rec = {"text": "Warfarin and aspirin increase bleeding; aspirin alone is safe.",
           "e1": {"text": "Warfarin", "offsets": [[0, 7]]},
           "e2": {"text": "aspirin", "offsets": [[13, 19]]}}
    assert mark_entities(rec) == "<D1>Warfarin</D1> and <D2>aspirin</D2> increase bleeding; aspirin alone is safe."
    # stessa parola due volte: si marca l'occorrenza giusta (la seconda), non la prima
    rec["e2"]["offsets"] = [[36, 42]]
    assert mark_entities(rec).count("<D2>") == 1 and "aspirin increase" in mark_entities(rec)
    # intervalli sovrapposti -> frase senza tag
    rec2 = {"text": "diagnostic or therapeutic monoclonal antibodies",
            "e1": {"text": "diagnostic monoclonal antibodies", "offsets": [[0, 9], [26, 46]]},
            "e2": {"text": "therapeutic monoclonal antibodies", "offsets": [[14, 46]]}}
    assert "<D1>" not in mark_entities(rec2)
    # su TUTTO il train: nessun errore, e tag sempre bilanciati dove presenti
    n = 0
    for line in open(PROCESSED_DIR / "train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        s = mark_entities(r)
        assert s.count("<D1>") == s.count("</D1>") and s.count("<D2>") == s.count("</D2>")
        n += 1
    ok(f"marcatura entita' corretta (controllate {n} coppie del train)")


def test_parsing():
    assert parse_prediction('{"label": "effect"}')["label"] == "effect"
    assert parse_prediction('```json\n{"label": "Advise"}\n```')["label"] == "advise"
    assert parse_prediction('Sure! {"label": "mechanism", "rationale": "x"} Hope it helps')["label"] == "mechanism"
    assert parse_prediction('{"label": "advice"}')["label"] == "advise"      # sinonimo
    assert parse_prediction('{"label": "no interaction"}')["label"] == "none"  # sinonimo
    for bad in ["", "boh", '{"label": "banana"}', '{"x": 1}', "[1,2]"]:
        assert parse_prediction(bad)["label"] == INVALID_LABEL, bad
    ok("parsing dell'output del modello (JSON, code fence, testo extra, sinonimi, errori)")


class FakeClient:
    """Finto modello: restituisce risposte scriptate, o solleva errori scriptati."""
    def __init__(self, script):
        self.script, self.n = list(script), 0

    def invoke(self, prompt):
        self.n += 1
        item = self.script.pop(0) if self.script else '{"label": "none"}'
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item)


def test_llm_wrapper(tmp):
    sleeps = []
    c = FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED rate limit"), '{"label": "effect"}'])
    llm = CachedLLM("modello-test", rpm=60, cache_dir=tmp, client=c, sleep=sleeps.append)
    assert llm.complete("p1") == '{"label": "effect"}' and c.n == 2   # un retry
    assert len(sleeps) >= 1
    assert llm.complete("p1") == '{"label": "effect"}' and c.n == 2   # dalla cache, nessuna chiamata
    assert llm.cache_hits == 1
    # la cache sopravvive a un nuovo oggetto (cioe' a un nuovo avvio del programma)
    llm2 = CachedLLM("modello-test", rpm=60, cache_dir=tmp, client=FakeClient([]), sleep=sleeps.append)
    assert llm2.complete("p1") == '{"label": "effect"}' and llm2.calls == 0
    # risposta a blocchi (lista) invece di stringa
    llm3 = CachedLLM("m3", rpm=60, cache_dir=tmp, client=FakeClient([[{"type": "text", "text": '{"label": "int"}'}]]),
                     sleep=sleeps.append)
    assert llm3.complete("q") == '{"label": "int"}'
    # quota giornaliera: niente retry, eccezione dedicata
    llm4 = CachedLLM("m4", rpm=60, cache_dir=tmp,
                     client=FakeClient([RuntimeError("429 Quota exceeded for metric ... PerDay")]), sleep=sleeps.append)
    try:
        llm4.complete("z")
        raise AssertionError("doveva sollevare DailyQuotaExceeded")
    except DailyQuotaExceeded:
        pass
    # errore non temporaneo: non si ritenta
    c5 = FakeClient([ValueError("API key not valid")])
    llm5 = CachedLLM("m5", rpm=60, cache_dir=tmp, client=c5, sleep=sleeps.append)
    try:
        llm5.complete("y")
        raise AssertionError("doveva propagare l'errore")
    except ValueError:
        assert c5.n == 1
    ok("client LLM: cache, retry su 429, quota giornaliera, errori non temporanei, risposte a blocchi")


def test_network_errors(tmp):
    import httpx
    sleeps = []
    c = FakeClient([httpx.ReadError("[Errno 54] Connection reset by peer"),
                    httpx.ConnectError("All connection attempts failed"),
                    '{"label": "advise"}'])
    llm = CachedLLM("m-net", rpm=60, cache_dir=tmp, client=c, sleep=sleeps.append)
    assert llm.complete("n1") == '{"label": "advise"}' and c.n == 3
    ok("errori di rete (connessione interrotta, stop del computer) ritentati")


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


FAKE_RX = {
    "warfarin": ["11289"], "Coumadin": ["202421"], "warfarin sodium": ["114248"],
    "aspirin": ["1191"], "Bactrim": ["B1"], "trimethoprim": ["T1"],
}
FAKE_PROPS = {
    "11289": ("warfarin", "IN"), "202421": ("Coumadin", "BN"), "114248": ("warfarin sodium", "PIN"),
    "1191": ("aspirin", "IN"), "B1": ("Bactrim", "BN"), "T1": ("trimethoprim", "IN"),
}
FAKE_ING = {"202421": [("11289", "warfarin")], "114248": [("11289", "warfarin")],
            "B1": [("S1", "sulfamethoxazole"), ("T1", "trimethoprim")]}


class FakeRxSession:
    """Finto server RxNorm con poche voci; replica la forma delle risposte reali."""
    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(url)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        params = params or {}
        if url.endswith("/rxcui.json"):
            ids = FAKE_RX.get(params["name"])
            return FakeResponse(200, {"idGroup": {"name": params["name"], **({"rxnormId": ids} if ids else {})}})
        if url.endswith("/properties.json"):
            c = url.split("/rxcui/")[1].split("/")[0]
            name, tty = FAKE_PROPS[c]
            return FakeResponse(200, {"properties": {"rxcui": c, "name": name, "tty": tty}})
        if url.endswith("/related.json"):
            c = url.split("/rxcui/")[1].split("/")[0]
            props = [{"rxcui": r, "name": n, "tty": "IN"} for r, n in FAKE_ING.get(c, [])]
            return FakeResponse(200, {"relatedGroup": {"rxcui": c, "conceptGroup": [{"tty": "IN", "conceptProperties": props}]}})
        return FakeResponse(404, None)


def test_rxnorm(tmp):
    import requests
    from src.tools.rxnorm import RxNormClient, RxNormError, same_drug

    path = tmp / "rx.jsonl"
    sess = FakeRxSession()
    c = RxNormClient(cache_path=path, rps=0, session=sess, sleep=lambda s: None)
    assert same_drug("Coumadin", "warfarin", c) == (True, "rxnorm")           # nome commerciale = generico
    assert same_drug("warfarin sodium", "warfarin", c) == (True, "rxnorm")    # sale = principio attivo
    assert same_drug("Warfarin ", "warfarin", c) == (True, "string")          # solo maiuscole/spazi
    assert same_drug("warfarin", "aspirin", c) == (False, "different")
    assert same_drug("quinolones", "warfarin", c) == (False, "unresolved")    # classe: non riconosciuta
    # prudenza: un prodotto combinato NON e' uno dei suoi componenti
    assert same_drug("Bactrim", "trimethoprim", c) == (False, "different")
    # cache su disco: un nuovo client non rifa' nessuna chiamata HTTP
    sess2 = FakeRxSession()
    c2 = RxNormClient(cache_path=path, rps=0, session=sess2, sleep=lambda s: None)
    assert same_drug("Coumadin", "warfarin", c2) == (True, "rxnorm") and c2.http_calls == 0 and sess2.calls == []

    # retry: 429, poi errore di rete, poi risposta valida
    ok = FakeResponse(200, {"idGroup": {"name": "x", "rxnormId": ["11289"]}})
    sess3 = FakeRxSession([FakeResponse(429), requests.ConnectionError("giu'"), ok])
    c3 = RxNormClient(cache_path=tmp / "rx3.jsonl", rps=0, session=sess3, sleep=lambda s: None)
    assert c3.rxcuis_by_name("x") == ["11289"] and c3.http_calls == 3
    # 404 = "niente", non un errore
    sess4 = FakeRxSession([FakeResponse(404)])
    c4 = RxNormClient(cache_path=tmp / "rx4.jsonl", rps=0, session=sess4, sleep=lambda s: None)
    assert c4.rxcuis_by_name("boh") == []
    # rete sempre giu': errore chiaro dopo i tentativi, nulla viene messo in cache
    sess5 = FakeRxSession([requests.ConnectionError("giu'")] * 5)
    c5 = RxNormClient(cache_path=tmp / "rx5.jsonl", rps=0, session=sess5, sleep=lambda s: None, max_attempts=5)
    try:
        c5.rxcuis_by_name("y")
        raise AssertionError("doveva fallire")
    except RxNormError:
        assert not (tmp / "rx5.jsonl").exists() or (tmp / "rx5.jsonl").read_text() == ""
    ok_("client RxNorm: nome commerciale/sale/generico, prudenza sui combinati, cache, retry, 404, errori")


def test_study_plumbing(tmp):
    import scripts.study_same_drug as study
    from src.tools.rxnorm import RxNormClient

    study.RESULTS_DIR = tmp
    study.RxNormClient = lambda: RxNormClient(cache_path=tmp / "rxs.jsonl", rps=0,
                                              session=FakeRxSession(), sleep=lambda s: None)
    import sys
    sys.argv = ["study", "--max-names", "40", "--none-sample", "300"]
    study.main()
    assert (tmp / "study_same_drug.txt").exists()
    ok_("script dello studio: esegue fino in fondo e scrive il rapporto (con server finto)")


class FakeLLM:
    """Finto CachedLLM: risposte scriptate (lista) o calcolate da una funzione del prompt."""
    def __init__(self, responses, name="fake"):
        self.responses, self.model_name = responses, name
        self.calls, self.cache_hits, self.prompts = 0, 0, []

    def complete(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        return self.responses(prompt) if callable(self.responses) else self.responses.pop(0)


def _rec(a="Warfarin", b="aspirin", label="effect"):
    return {"pair_id": f"t.{a}.{b}", "doc_id": "d", "sentence_id": "s", "source": "DrugBank",
            "text": f"{a} and {b} increase bleeding.", "label": label,
            "e1": {"id": "e0", "text": a, "type": "drug", "offsets": [[0, len(a) - 1]]},
            "e2": {"id": "e1", "text": b, "type": "drug", "offsets": [[len(a) + 5, len(a) + 4 + len(b)]]}}


def _lab(x):
    return json.dumps({"label": x})


def _verdict(v, issue="none", critique=""):
    return json.dumps({"verdict": v, "issue": issue, "critique": critique})


def test_graph(tmp):
    from src.graph import build_agent
    from src.prompts import build_baseline_prompt
    from src.tools.rxnorm import RxNormClient

    fs = []
    rx = RxNormClient(cache_path=tmp / "rxg.jsonl", rps=0, session=FakeRxSession(), sleep=lambda s: None)

    def run(ext, jud, rec=None, rxc=None, **kw):
        e, j = FakeLLM(list(ext)), FakeLLM(list(jud))
        out = build_agent(e, j, fs, rxc, **kw)(rec or _rec())
        return out, e, j

    # 1. stesso farmaco (nome commerciale/generico): nessuna chiamata ai modelli
    out, e, j = run([], [], _rec("Coumadin", "warfarin"), rx)
    assert out["pred"] == "none" and out["status"] == "identity" and e.calls == 0 and j.calls == 0
    # 2. l'estrattore dice none: il judge non viene chiamato
    out, e, j = run([_lab("none")], [])
    assert out["pred"] == "none" and out["status"] == "accepted_none" and j.calls == 0
    # 3. interazione accettata dal judge
    out, e, j = run([_lab("effect")], [_verdict("accept")])
    assert (out["pred"], out["status"], out["attempts"], out["judge_calls"]) == ("effect", "accepted", 1, 1)
    # 4. rifiuto -> critica passata all'estrattore -> corregge in none
    out, e, j = run([_lab("effect"), _lab("none")], [_verdict("reject", "not_between_pair", "They are only listed.")])
    assert "Reviewer's comment" not in e.prompts[0] and "They are only listed." in e.prompts[1]
    assert (out["pred"], out["status"], out["attempts"], out["judge_calls"]) == ("none", "accepted_none", 2, 1)
    # 5. stallo: l'estrattore ripete la risposta -> incerto, senza richiamare il judge
    out, e, j = run([_lab("mechanism"), _lab("mechanism")], [_verdict("reject", "wrong_type", "x")])
    assert out["status"] == "uncertain" and out["pred"] == "none" and out["uncertain_label"] == "mechanism" and j.calls == 1
    out, _, _ = run([_lab("mechanism"), _lab("mechanism")], [_verdict("reject")], fallback="keep")
    assert out["pred"] == "mechanism" and out["uncertain_label"] == "mechanism"
    # 6. tentativi esauriti (3) con judge che rifiuta sempre; e veto con un solo tentativo
    out, e, j = run([_lab("effect"), _lab("mechanism"), _lab("advise")], [_verdict("reject")] * 3, max_attempts=3)
    assert out["status"] == "uncertain" and out["attempts"] == 3 and j.calls == 3 and out["uncertain_label"] == "advise"
    out, e, j = run([_lab("effect")], [_verdict("reject")], max_attempts=1)
    assert out["status"] == "uncertain" and out["pred"] == "none" and e.calls == 1
    # 7. output del judge illeggibile: nessun veto, ma segnalato
    out, _, _ = run([_lab("effect")], ["boh"])
    assert out["pred"] == "effect" and out["status"] == "accepted" and out["judge_parse_failures"] == 1
    # 8. output illeggibile dell'estrattore: INVALID, nessuna chiamata al judge
    out, e, j = run(["boh"], [])
    assert out["pred"] == "INVALID" and out["status"] == "invalid" and j.calls == 0 and out["parse_ok"] is False
    # 9. senza judge
    e = FakeLLM([_lab("advise")])
    out = build_agent(e, None, fs, None, use_judge=False)(_rec())
    assert out["pred"] == "advise" and out["status"] == "no_judge"
    # 10. judge_scope="all": il judge controlla anche i none
    out, e, j = run([_lab("none")], [_verdict("accept")], judge_scope="all")
    assert j.calls == 1
    # 11. il primo tentativo ha ESATTAMENTE il prompt della baseline (stessa chiave di cache)
    out, e, j = run([_lab("effect")], [_verdict("accept")])
    assert e.prompts[0] == build_baseline_prompt(_rec(), fs)
    ok_("grafo: identita', none senza judge, accetta, rifiuta+critica, stallo, tentativi esauriti, veto, "
        "output illeggibili, prompt del primo tentativo = baseline")


def test_run_eval_agent(tmp):
    run_eval.RESULTS_DIR = tmp
    ext = FakeLLM(lambda p: _lab("effect"), name="ext")
    jud = FakeLLM(lambda p: _verdict("accept"), name="jud")
    from src.tools.rxnorm import RxNormClient
    rx = RxNormClient(cache_path=tmp / "rxr.jsonl", rps=0, session=FakeRxSession(), sleep=lambda s: None)
    m = run_eval.main(["--config", "agent", "--max-attempts", "3", "--split", "dev_sample", "--shots", "1",
                       "--limit", "6"], llm=ext, judge_llm=jud, rx_client=rx)
    assert (tmp / "agent3_s1_dev_sample.jsonl").exists() and (tmp / "agent3_s1_dev_sample_metrics.json").exists()
    assert "agent" in m and m["run"]["config"] == "agent3" and m["run"]["judge_model"] == "jud"
    assert "judge_vs_gold_first_attempt" in m["agent"]
    ok_("run_eval con la configurazione agent: file, metriche e diagnostica del judge")


def test_rate_limit(tmp):
    t = {"now": 0.0}
    sleeps = []
    def sleep(s): sleeps.append(s); t["now"] += s
    llm = CachedLLM("m-rl", rpm=6, cache_dir=tmp, client=FakeClient([]), sleep=sleep, clock=lambda: t["now"])
    llm.complete("a"); llm.complete("b")
    assert abs(sleeps[0] - 10.0) < 1e-6, sleeps   # 6 RPM = una chiamata ogni 10 s
    ok("rate limit: 6 richieste/minuto -> 10 s tra due chiamate")


def test_run_eval_resume(tmp):
    run_eval.RESULTS_DIR = tmp
    # 1a esecuzione: il modello "esaurisce la quota" dopo 3 coppie
    script = ['{"label": "none"}'] * 3 + [RuntimeError("429 ... PerDay")]
    llm = CachedLLM("m-run", rpm=0, cache_dir=tmp, client=FakeClient(script), sleep=lambda s: None)
    try:
        run_eval.main(["--config", "baseline", "--split", "dev_sample", "--shots", "1", "--limit", "8"], llm=llm)
        raise AssertionError("doveva fermarsi")
    except SystemExit as e:
        assert e.code == 2
    out = tmp / "baseline_s1_dev_sample.jsonl"
    assert len(out.read_text().splitlines()) == 3
    # 2a esecuzione (giorno dopo): riprende e completa solo le 5 mancanti
    llm2 = CachedLLM("m-run", rpm=0, cache_dir=tmp, client=FakeClient([]), sleep=lambda s: None)
    m = run_eval.main(["--config", "baseline", "--split", "dev_sample", "--shots", "1", "--limit", "8"], llm=llm2)
    assert len(out.read_text().splitlines()) == 8 and llm2.calls == 5 and m["n"] == 8
    assert (tmp / "baseline_s1_dev_sample_metrics.json").exists()
    ok("run_eval: si ferma sulla quota giornaliera e al rilancio riprende senza rifare nulla")


def test_prompt():
    train = [json.loads(l) for l in open(PROCESSED_DIR / "train.jsonl", encoding="utf-8")]
    dev = [json.loads(l) for l in open(PROCESSED_DIR / "dev_sample.jsonl", encoding="utf-8")]
    fs = select_fewshot(train, per_label=2)
    assert len(fs) == 10 and fs == select_fewshot(train, per_label=2)          # deterministico
    assert {r["doc_id"] for r in fs}.isdisjoint({r["doc_id"] for r in dev})    # nessun leakage
    p = build_baseline_prompt(dev[0], fs)
    assert "<D1>" in p or "Drug 1:" in p and p.rstrip().endswith("Answer:")
    ok(f"few-shot: 10 esempi deterministici, nessun documento in comune con dev (prompt di {len(p)} caratteri)")


if __name__ == "__main__":
    test_marking()
    test_parsing()
    test_prompt()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        test_llm_wrapper(d / "a")
        test_rate_limit(d / "b")
        test_network_errors(d / "d")
        test_rxnorm(d / "e")
        test_study_plumbing(d / "f")
        test_graph(d / "g")
        test_run_eval_agent(d / "h")
        test_run_eval_resume(d / "c")
    print("\nTutti i test offline sono passati.")
