CORPUS=data/raw/corpus.arrow
DIRECTION=combinatorics
JUDGES=1

LABEL_MODEL=gpt-oss-120b
LABEL_API=chat
LABEL_JOBS=128
LABEL_RAMP=0

EXTRACT_MODEL=gemini-3.5-flash
EXTRACT_API=chat
EXTRACT_JOBS=96
EXTRACT_RAMP=0

CHECK_MODEL=gemini-3.1-pro
CHECK_API=chat
CHECK_WEB_SEARCH=true
CHECK_JOBS=64
CHECK_RAMP=0

SOLVE_MODEL=gpt-5.5
SOLVE_EFFORT=xhigh
SOLVE_JOBS=64
SOLVE_RAMP=0

JUDGE_MODEL=gpt-5.5
JUDGE_EFFORT=xhigh
JUDGE_JOBS=64
JUDGE_RAMP=0

GRADE_MODEL=gpt-5.5
GRADE_EFFORT=xhigh
GRADE_JOBS=64
GRADE_RAMP=0

python src/pipeline.py \
    --stage all \
    --corpus "$CORPUS" \
    --direction "$DIRECTION" \
    --judges "$JUDGES" \
    --label-model "$LABEL_MODEL" \
    --label-api "$LABEL_API" \
    --label-jobs "$LABEL_JOBS" \
    --label-ramp "$LABEL_RAMP" \
    --extract-model "$EXTRACT_MODEL" \
    --extract-api "$EXTRACT_API" \
    --extract-jobs "$EXTRACT_JOBS" \
    --extract-ramp "$EXTRACT_RAMP" \
    --check-model "$CHECK_MODEL" \
    --check-api "$CHECK_API" \
    --check-web-search "$CHECK_WEB_SEARCH" \
    --check-jobs "$CHECK_JOBS" \
    --check-ramp "$CHECK_RAMP" \
    --solve-model "$SOLVE_MODEL" \
    --solve-effort "$SOLVE_EFFORT" \
    --solve-jobs "$SOLVE_JOBS" \
    --solve-ramp "$SOLVE_RAMP" \
    --judge-model "$JUDGE_MODEL" \
    --judge-effort "$JUDGE_EFFORT" \
    --judge-jobs "$JUDGE_JOBS" \
    --judge-ramp "$JUDGE_RAMP" \
    --grade-model "$GRADE_MODEL" \
    --grade-effort "$GRADE_EFFORT" \
    --grade-jobs "$GRADE_JOBS" \
    --grade-ramp "$GRADE_RAMP" \
    > run_all.log 2>&1
