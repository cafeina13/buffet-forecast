# Site and model audits

Method taken from the student's capstone audits: the public site (Turkish), the README and
the code's claims are checked against this repository until an audit finds nothing. The
auditor is an independent agent that knows nothing of the conversation; Claude verifies
each finding before changing anything, and fixes are applied only after all parts report.

| Folder | What |
| --- | --- |
| `exports/` | The site pages, their scripts and the README as audited (dated, frozen) |
| `findings/` | One file per audit: the findings, Claude's verdict on each, what was changed |
| `scripts/` | The auditors' own recomputation scripts (read-only checks) |
| `sources/` | Text of the cited sources, for checking claims (not in git: copyrighted) |

Give the auditor nothing about earlier audits: not their findings, counts, fixes or
rejected points, and not this folder's `findings/`. There is no target or limit on the
number of findings. Run one agent at a time (the laptop overheats).

## Parts

| Part | Scope |
| --- | --- |
| 1 | Home page (`index.html`) and its reference list |
| 2 | Chef's screen (`demo.html`, `js/demo.js`, `js/model.js`): numbers, explanations, tags, controls, monthly report |
| 3 | "Nasıl çalışır" (`nasil.html`, `js/nasil.js`) and the README: charts, metrics, method, tests |

Brief: `brief.md`.

| Audit | Date | Auditor | Result |
| --- | --- | --- | --- |
| 1 | 2026-10-24 | three partitioned agents, no hints | 43 findings, 4 errors (explanations, new-dish claim, guest drivers, embedding claim); fixed: `findings/audit1.md` |
| 2 | 2026-10-24 | three partitioned agents, no hints | 35 findings, 2 errors (both introduced by audit-1 fixes); fixed: `findings/audit2.md` |
| 3 | 2026-10-25 | three partitioned agents, no hints | running |
