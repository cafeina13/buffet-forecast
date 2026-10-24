# Brief for one part of a partitioned audit (used verbatim)

Replace {EXPORT} (e.g. 2026-10-24-audit1), {N} (audit number), {P} (part) and {SCOPE}.
Scopes:
- 1: the home page, site/index.html, including its reference list ("Kaynakça").
- 2: the chef's screen, site/demo.html with site/js/demo.js and site/js/model.js: every number it shows, the Turkish explanation sentences, the differentiator ("Özgünlük") tags, the scenario controls and the monthly report.
- 3: the "how it works" page, site/nasil.html with site/js/nasil.js, and README.md: the charts, the metrics, the description of the method and the tests.

---
You are auditing a student team's competition prototype, independently and read-only. It is a website (Turkish) that presents an AI tool forecasting how much of each dish to cook at all-inclusive hotel buffets, plus the Python code and synthetic data behind it. The repository is C:\Users\ZERO\Documents\GitHub\buffet-forecast (Windows; use the Bash tool with Unix syntax; Python via `uv run python`, run with PYTHONIOENCODING=utf-8; Node is available as `node`).

## What to audit
- audit/exports/{EXPORT}/site/: the site's three pages and their scripts, as published. The pages load JSON from web/data/ (demo.json, model-guests.json, model-dishes.json); those files are the published data.
- audit/exports/{EXPORT}/README.md: the repository README.

## Your part
The audit is split into three fixed parts, each checked by its own auditor. Yours: {SCOPE} You may read the rest of the site and repository for context, but report findings only in your part.

## How
Check every claim, number by number and sentence by sentence:
- Check each number's definition (what it counts and over what), not just its value. "Right number, wrong label" errors are the kind most likely to remain.
- Recompute numbers yourself from the code, from web/data/*.json, from experiments/*/metrics.json and from data/ (the generated synthetic tables), with your own code, not from text in the repository. Where the page computes something in JavaScript, you may run that code (for example with Node, loading web/js/model.js and the JSON files).
- Check claims about cited sources against the source texts in audit/sources/ (see its README.txt for what is and is not available), and claims about the code against the code.
- Check the Turkish text for meaning (not style): does each sentence say what the data and code support?
- The prototype runs only on synthetic or semi-synthetic data (some public real inputs; dish demand is always simulated). Check that nothing on the page presents a synthetic result as a measured one, and that the page does not name any real hotel, brand or person except as a cited source.
- Do not read audit/README.md, audit/findings/, audit/scripts/ (except your own folder), or other folders in audit/exports/. The two .md files in the repository root (a brief and a project draft) are planning notes, not evidence, and not under audit. Judge the work on your own.

## Rules
- Read-only: do not modify any file outside audit/scripts/audit{N}/part{P}/, where your own scratch scripts go.
- No model training, no long CPU load (the laptop overheats; a scoring pass of a minute or two is fine), no network, no installs.
- There is no target or limit on the number of findings: report every real one, however many or few. Do not pad with trivia, and do not stop early.

## Output (final message)
1. Findings, most serious first. For each: location (page and section, or file and line), the exact text, what is wrong, the evidence (file, values, your script), and a suggested fix. Class each as ERROR, MISLABEL, OVERSTATEMENT, INCONSISTENCY or MINOR.
2. What you checked and found correct, and how.
3. What you could not check, and why.
