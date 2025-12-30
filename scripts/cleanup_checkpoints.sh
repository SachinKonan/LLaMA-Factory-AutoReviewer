#!/bin/bash
# Auto-cleanup script for pytorch_model_fsdp.bin and optimizer.bin files
# Usage: nohup ./scripts/cleanup_checkpoints.sh &

WATCH_DIR="/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/saves/qwen2.5-7b/full/grid_searchv2"
LOG_FILE="/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/cleanup.log"
INTERVAL=300  # Check every 5 minutes

mkdir -p "$(dirname "$LOG_FILE")"

echo "$(date): Cleanup watcher started for $WATCH_DIR" >> "$LOG_FILE"
echo "$(date): Checking every ${INTERVAL}s" >> "$LOG_FILE"

while true; do
    # Find and delete files
    deleted=$(find "$WATCH_DIR" \( -name "pytorch_model_fsdp.bin" -o -name "optimizer.bin" \) -delete -print 2>/dev/null)

    if [ -n "$deleted" ]; then
        count=$(echo "$deleted" | wc -l)
        echo "$(date): Deleted $count files" >> "$LOG_FILE"
        echo "$deleted" >> "$LOG_FILE"
    fi

    sleep "$INTERVAL"
done
