# Known Failures

Before attempting a third fix for the same failure, check this file and apply any
relevant confirmed fix first.

## Policy

- If a failure occurs more than twice in a row, stop retrying the same path.
- Before a third attempt, check this file for prior solutions.
- If the failure is not documented, add an entry before continuing.
- Include the command, error text, context, attempts, why they failed, and next
  hypotheses.
- Once resolved, update the same entry with the confirmed fix and verification
  steps.
- Do not reapply a previously failed fix unless new evidence justifies it.

## Entry Template

```md
## <failure signature>

- First seen:
- Command:
- Context:
- Error text:
- Attempts already tried:
- Why attempts failed:
- Next hypotheses:
- Confirmed fix:
- Verification:
- Related files:
```

## Current Known Failures

No confirmed recurring failures have been recorded yet.
