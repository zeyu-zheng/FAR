CORPUS=data/raw/corpus.arrow
INPUT=data/checked.jsonl

SOLVE_MODEL=gpt-5.5
SOLVE_EFFORT=xhigh
SOLVE_JOBS=64
SOLVE_RAMP=0

python src/pipeline.py \
    --stage attempt \
    --corpus "$CORPUS" \
    --input "$INPUT" \
    --solve-model "$SOLVE_MODEL" \
    --solve-effort "$SOLVE_EFFORT" \
    --solve-jobs "$SOLVE_JOBS" \
    --solve-ramp "$SOLVE_RAMP" \
    > attempt.log 2>&1
