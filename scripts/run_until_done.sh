#!/bin/bash
# Rilancia un comando di valutazione finche' non termina bene: ogni run riprende sempre da dove si
# era fermata (le predizioni gia' fatte sono salvate), quindi un errore temporaneo di Gemini
# (503 "high demand") non fa perdere lavoro. NON modifica il codice del progetto.
#
# Uso:  scripts/run_until_done.sh python -m eval.run_eval --config agent --split test_sample ...
#
# Si ferma subito con codice 0 (finito), 2 (quota giornaliera esaurita: riprovare domani) o
# 130 (Ctrl+C). Per ogni altro errore riprova, al massimo 10 volte, attendendo 5 minuti
# (variabile RETRY_WAIT in secondi per cambiare l'attesa).
WAIT=${RETRY_WAIT:-300}
for i in $(seq 1 10); do
  echo ">>> tentativo $i di 10"
  caffeinate -i "$@"
  code=$?
  if [ $code -eq 0 ] || [ $code -eq 2 ] || [ $code -eq 130 ]; then
    exit $code
  fi
  echo ">>> terminato con codice $code: riprovo tra ${WAIT}s (Ctrl+C per fermare)"
  sleep "$WAIT"
done
echo ">>> troppi tentativi falliti"
exit 1
