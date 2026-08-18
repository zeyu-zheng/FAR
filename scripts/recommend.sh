CORPUS=data/raw/corpus.arrow
INPUT=data/solved.jsonl
JUDGES=1

JUDGE_MODEL=gpt-5.5
JUDGE_EFFORT=xhigh
JUDGE_JOBS=64
JUDGE_RAMP=0

GRADE_MODEL=gpt-5.5
GRADE_EFFORT=xhigh
GRADE_JOBS=64
GRADE_RAMP=0

python src/pipeline.py \
    --stage recommend \
    --corpus "$CORPUS" \
    --input "$INPUT" \
    --judges "$JUDGES" \
    --judge-model "$JUDGE_MODEL" \
    --judge-effort "$JUDGE_EFFORT" \
    --judge-jobs "$JUDGE_JOBS" \
    --judge-ramp "$JUDGE_RAMP" \
    --grade-model "$GRADE_MODEL" \
    --grade-effort "$GRADE_EFFORT" \
    --grade-jobs "$GRADE_JOBS" \
    --grade-ramp "$GRADE_RAMP" \
    > recommend.log 2>&1
