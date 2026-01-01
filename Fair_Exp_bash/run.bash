TOTAL_TIME=$((30))  # 30 seconds

timeout --signal=SIGTERM --kill-after=3s ${TOTAL_TIME}s \
python test.py


TOTAL_TIME=$((103 * 150)) # toal seconds
echo "==== Starting inference with time budget of ${TOTAL_TIME} seconds... ===="
timeout --signal=SIGTERM --kill-after=3s ${TOTAL_TIME}s \
python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE 


