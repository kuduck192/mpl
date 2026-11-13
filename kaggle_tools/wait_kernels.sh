#!/usr/bin/env bash
# Poll Kaggle kernel statuses gently (default every 300 s); print a line on every change;
# exit once every kernel is in a terminal state (COMPLETE / ERROR / CANCEL*).
# Transient API / proxy errors are reported as "?" and do not end the wait.
#   kaggle_tools/wait_kernels.sh slug1 [slug2 ...]
cd "$(dirname "$0")/.." || exit 1
INTERVAL=${INTERVAL:-300}
prev=""
while true; do
  cur=""
  done_all=1
  for s in "$@"; do
    st=$(python3 kaggle_tools/kaggle_run.py status "$s" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1)
    st=${st#KernelWorkerStatus.}
    [ -z "$st" ] && st="?"
    case "$st" in COMPLETE|ERROR|CANCEL*) ;; *) done_all=0 ;; esac
    cur="$cur $s:$st"
  done
  if [ "$cur" != "$prev" ]; then echo "$(date -u +%H:%M)$cur"; prev="$cur"; fi
  [ "$done_all" = 1 ] && { echo "all terminal:$cur"; break; }
  sleep "$INTERVAL"
done
