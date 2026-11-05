# Agent-00Pharma

A [LangGraph](https://github.com/langchain-ai/langgraph) agent for **drug–drug interaction (DDI) classification** on the [DDI Corpus](https://github.com/isegura/DDICorpus) (SemEval-2013 Task 9.2). Given a sentence with two already-annotated drug mentions, the system decides whether they interact and, if so, which of four types: `effect`, `mechanism`, `advise`, `int` (or `none`).

> [!NOTE]
> University project for the Natural Language Processing course (MSc Computer Science, A.Y. 2026/2027). Not intended for medical use: do not rely on it for medical advice.

## How it works

1. **Identity check** (RxNorm): if the two mentions refer to the same drug, the pair is labelled `none` without calling any model.
2. **Extractor** (LLM, `gemini-3.5-flash-lite`): proposes a label (few-shot prompt, JSON output).
3. **Judge** (LLM, `gemini-3.1-flash-lite`): accepts or rejects the label and, on rejection, returns a critique; the extractor retries, up to 3 attempts.
4. **Uncertain node**: if the judge still rejects, a policy decides: `veto` → `none`, `keep` → the last label proposed by the extractor.

The judge can check only interaction labels or also `none` labels, and uses one of two rubrics (v1, v2). Configurations are cumulative: `baseline` → `identity` → `judge` → `agent`.

## Results at a glance

The main hypothesis (the full agent beats the single-extractor reference) is **not confirmed**:

- 204 → 210 correct labels out of 260 (10 fixed / 4 broken, sign test p = 0.18).
- A cautious judge used as a simple veto reduces false positives with negligible recall loss (exploratory); a confirmation on a second sample did not meet the pre-registered criterion (Δ detection F1 reweighted = +0.033 ± 0.021, 95% CI [−0.004, +0.079]).
- The critique loop and the `none` check add nothing over the veto alone.

Evaluation uses small stratified samples, paired comparisons, sentence-level bootstrap standard errors and a reweighted estimate at the real class prevalence. See the report for details and limitations.

## Repository structure

```
config.py     # paths, models, seeds
src/          # agent (LangGraph graph, prompts, LLM and RxNorm clients)
eval/         # evaluation: runs, tables, paired tests, bootstrap, ablation, error analysis
scripts/      # sampling, figures, helper scripts
data/         # (not versioned) raw and processed corpus
results/      # (not versioned) predictions and tables
report/       # final report (Italian)
```

## Setup

Developed on Python 3.14.7.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**API key.** Create a free key in [Google AI Studio](https://aistudio.google.com/) and put it in a `.env` file at the repository root:

```
GOOGLE_API_KEY=your_key_here
```

## Data

The DDI Corpus is not redistributed here. Download it (XML version) from the [original repository](https://github.com/isegura/DDICorpus) (CC BY-NC) and place it under `data/raw/`:

```bash
mkdir -p data/raw
git clone https://github.com/isegura/DDICorpus.git data/raw/DDICorpus
```

### Preprocessing

The preprocessing step parses the XML files and builds one JSONL record per drug pair, then draws the evaluation samples:

```bash
python -m <preprocessing_module>        # parsing + train/dev/test split
python -m <sampling_module>             # dev_sample, test_sample
python -m scripts.make_confirm_sample   # test_confirm (sentence-disjoint from test_sample)
```

What it does:

- parses each document (sentences, drug mentions with text, type and character offsets, annotated pairs with their label);
- builds one record per pair, with the two mentions marked by position as `<D1>...</D1>` and `<D2>...</D2>`, so that repeated mentions of the same drug are unambiguous;
- keeps the official train/test split and derives a development set (`dev`) from the training set by assigning whole documents, so sentences from the same document never end up in different sets;
- draws class-stratified samples with fixed seeds (42; 2026 for `test_confirm`), because the free tier of the models limits how many pairs can be evaluated.

Outputs are written to `data/processed/` as `.jsonl` files. Expected sizes, as a sanity check:

| Set | Pairs | Notes |
|---|---|---|
| `train` | 25,719 | 562 documents |
| `dev` | 2,072 | 62 documents, taken from the training set |
| `test` | 5,716 | 191 documents, official test set |
| `dev_sample` | 105 | 40 `none`, 65 with interaction |
| `test_sample` | 260 | 100 `none`, 160 with interaction |
| `test_confirm` | 225 | 100 `none`, 125 with interaction |

## Usage

Run an evaluation (responses are cached on disk, so repeated runs cost nothing):

```bash
python -m eval.run_eval --config judge --judge-version 2 --split dev_sample --shots 2
python -m eval.run_eval --help     # all options (agent configurations, scope, policy, ...)
```

Build tables and statistics from the stored predictions (no API calls):

```bash
python -m eval.make_table --runs baseline_s2_test_sample identity_s2_test_sample judgev2_s2_test_sample --sd
python -m eval.compare_runs --a identity_s2_test_sample --b judgev2_s2_test_sample
python -m eval.reweight --runs identity_s2_test_sample judgev2_s2_test_sample --ref identity_s2_test_sample --sd
python -m eval.ablation
```

Run names follow `<config>_s<shots>_<split>` (e.g. `agent3allv2_s2_test_sample`: agent, 3 attempts, judge also on `none`, rubric v2). The free tier allows about 500 requests per day per model, so a full evaluation takes several days.

## Reproducibility

Fixed seeds (42; 2026 for the confirmation sample), pinned versions in `requirements.txt`, disk cache of every model and RxNorm response, and version tags `freeze-pre-test` and `pre-conferma` that fix the code used for the reported results. The models are not deterministic, so exact re-generation without the cache may produce different responses; run-to-run variability was not measured.

## Report

The full report (in Italian) is in [`report/`](report/).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). The DDI Corpus is distributed by its authors under CC BY-NC and is not included in this repository.

## References

- Herrero-Zazo et al., *The DDI corpus: an annotated corpus with pharmacological substances and drug–drug interactions*, J. Biomed. Inform. 46(5), 2013.
- Segura-Bedmar et al., *SemEval-2013 Task 9: Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)*, SemEval 2013.
