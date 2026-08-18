CORPUS=data/raw/corpus.arrow
DIRECTION=combinatorics

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

python src/pipeline.py \
    --stage find \
    --corpus "$CORPUS" \
    --direction "$DIRECTION" \
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
    > find.log 2>&1
