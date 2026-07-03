#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 START_DATE (format: YEAR-MM-DD) END_DATE [LABEL] [BRANCH1,BRANCH2] [LIMIT]"
    exit 1
fi

START_DATE="$1"
END_DATE="$2"

if [ -n "$4" ]; then
    IFS=',' read -ra BRANCHES <<< "$4"
else
    BRANCHES=("main")
fi

if [ -n "$5" ]; then
    LIMIT=$5
else
    LIMIT=300
fi

OUTPUT=""
for BRANCH in "${BRANCHES[@]}"; do
    OUTPUT+=$(gh pr list --repo SpikeInterface/probeinterface --limit $LIMIT  --base "$BRANCH" --state merged --json number,title,mergedAt \
        | jq -r --arg start_date "${START_DATE}T00:00:00Z" --arg end_date "${END_DATE}T00:00:00Z" \
        '.[] | select(.mergedAt >= $start_date and .mergedAt <= $end_date) | "* \(.title) (#\(.number))"')
done
if [ -n "$OUTPUT" ]; then
    echo ""
    echo "$OUTPUT"
    echo ""
fi

echo "Contributors:"
echo ""
gh pr list --repo SpikeInterface/probeinterface --limit 1000 --base main --state merged --json number,title,author,mergedAt \
  | jq -r --arg start_date "${START_DATE}T00:00:00Z" --arg end_date "${END_DATE}T00:00:00Z" \
  '[.[] | select(.mergedAt >= $start_date and .mergedAt <= $end_date and .author.login != "app/pre-commit-ci") | .author.login] | unique | .[] | "* @" + .'
