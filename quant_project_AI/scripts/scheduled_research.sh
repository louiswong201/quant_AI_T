#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  V4 Scheduled Strategy Research
#  Called by cron/launchd/Task Scheduler for automated research.
#
#  Usage:
#    ./scripts/scheduled_research.sh daily     # monitor only (~15s)
#    ./scripts/scheduled_research.sh weekly    # monitor + optimize + portfolio + anomaly
#    ./scripts/scheduled_research.sh monthly   # all engines, deep (EXPANDED grids)
#    ./scripts/scheduled_research.sh triggered # immediate optimize for ALERT symbols
#
#  Scheduling recommendations:
#    Daily   — 08:00     crontab: 0 8 * * * /path/to/scheduled_research.sh daily
#    Weekly  — Mon 07:00 crontab: 0 7 * * 1 /path/to/scheduled_research.sh weekly
#    Monthly — 1st 06:00 crontab: 0 6 1 * * /path/to/scheduled_research.sh monthly
# ─────────────────────────────────────────────────────────────

set -euo pipefail

MODE="${1:-daily}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON="python3"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/research_${MODE}_${TIMESTAMP}.log"

cd "$PROJECT_DIR"

echo "[$TIMESTAMP] Starting V4 $MODE research..." | tee "$LOG_FILE"

case "$MODE" in
  daily)
    $PYTHON daily_research.py --mode daily 2>&1 | tee -a "$LOG_FILE"
    ;;
  weekly)
    $PYTHON daily_research.py --mode weekly 2>&1 | tee -a "$LOG_FILE"
    ;;
  monthly)
    $PYTHON daily_research.py --mode monthly --apply 2>&1 | tee -a "$LOG_FILE"
    ;;
  triggered)
    $PYTHON daily_research.py --mode triggered 2>&1 | tee -a "$LOG_FILE"
    ;;
  *)
    echo "Unknown mode: $MODE. Use daily/weekly/monthly/triggered." | tee -a "$LOG_FILE"
    exit 1
    ;;
esac

EXIT_CODE=$?
END_TIME=$(date +%Y%m%d_%H%M%S)

echo "[$END_TIME] V4 $MODE research finished (exit=$EXIT_CODE)" | tee -a "$LOG_FILE"

# ── Check for ALERTs and notify ──
REPORT=$(ls -t "$PROJECT_DIR/reports/daily_research_"*.md 2>/dev/null | head -1)
if [ -n "$REPORT" ]; then
  ALERT_COUNT=$(grep -c "ALERT" "$REPORT" 2>/dev/null || true)
  HIGH_COUNT=$(grep -c "HIGH" "$REPORT" 2>/dev/null || true)

  if [ "$ALERT_COUNT" -gt 0 ] || [ "$HIGH_COUNT" -gt 0 ]; then
    # macOS notification
    osascript -e "display notification \"$ALERT_COUNT alerts, $HIGH_COUNT high-priority items\" with title \"V4 Strategy Research\" subtitle \"$MODE scan complete\"" 2>/dev/null || true
    # Windows notification (if powershell available)
    powershell.exe -Command "New-BurntToastNotification -Text 'V4 Strategy Research','$MODE: $ALERT_COUNT alerts, $HIGH_COUNT high-priority'" 2>/dev/null || true
    echo "WARNING: $ALERT_COUNT ALERTs, $HIGH_COUNT HIGH-priority items" | tee -a "$LOG_FILE"

    # Auto-trigger immediate optimize if daily mode found ALERTs
    if [ "$MODE" = "daily" ] && [ "$ALERT_COUNT" -gt 0 ]; then
      echo "Auto-triggering optimize for ALERT symbols..." | tee -a "$LOG_FILE"
      $PYTHON daily_research.py --mode triggered 2>&1 | tee -a "$LOG_FILE"
    fi
  else
    osascript -e "display notification \"All strategies stable\" with title \"V4 Strategy Research\" subtitle \"$MODE scan complete\"" 2>/dev/null || true
    powershell.exe -Command "New-BurntToastNotification -Text 'V4 Strategy Research','$MODE: All stable'" 2>/dev/null || true
  fi
fi

# ── Clean old logs (keep 30 days) ──
find "$LOG_DIR" -name "research_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
