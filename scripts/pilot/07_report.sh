#!/usr/bin/env bash
# Analysis CSVs (results/pilot-analysis/) and the labelled pilot report (dissertation/generated-pilot/).
# Never writes to dissertation/generated/.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 07_report
require_no_campaign

OUT=/work/dissertation/generated-pilot
sb local analyze analyze --matrix "$MATRIX" --db "$DB" --out results/pilot-analysis
sb local report report --matrix "$MATRIX" --db "$DB" --out "$OUT"
echo "report written to $PROJECT/dissertation/generated-pilot"
