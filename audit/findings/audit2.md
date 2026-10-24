# Audit 2 (2026-10-24): partitioned, three agents one at a time, no hints

Input frozen at `../exports/2026-10-24-audit2/`; brief `../brief.md`.
Fixes are applied only after all three parts report. DRAFT while parts run.

## Part 1: home page and references (10 findings)

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 1.1 | MISLABEL | Risk defined against "ilk pişirilen miktar"; the refill rule cooks ~40% first; the evaluation assumes the whole plan cooked then refilled up to +12% | Confirmed (rec = (1−risk) quantile of the whole amount; plan_short_rate = mean(y > rec)). Introduced by the audit-1 fix of 3.2. Same wording on the demo, nasıl and README | "Planlanan toplam miktar"; say the evaluation does not simulate the staggered refill timing |
| 1.2 | OVERSTATEMENT | "kazancın çoğu kuraldan geliyor" without the circularity caveat | Confirmed (generator eligible = present_guests) | Add the caveat |
| 1.3 | INCONSISTENCY | Monthly report "Önlenen atık" compares at the same risk setting, not at the same stock-out rate; uses simulated true demand, not observable in a kitchen | Confirmed | Card 4: pilot measures against baseline weeks; demo report is a simulation at a fixed setting |
| 1.4 | MISLABEL | "… 100 misafir başına ortalama 8,4 kg tüketildi" is the stock-out-corrected history | Confirmed (history_tables uses target_kg); 8.425 corrected vs 8.361 observed | Say "(biten servisler düzeltilerek)" here and in the demo sentence |
| 1.5 | MINOR | "etkisi henüz ölçülemedi" hides the synthetic comparison | Accepted | "sentetik testte belirgin fark görülmedi" |
| 1.6 | MINOR | Leftover measurement not exercised in the prototype | Confirmed (leftover_kg unused) | Say so |
| 1.7 | MINOR | "yönü literatürden" is plate-waste findings; Aydın gives importance, not direction | Accepted | Reword |
| 1.8 | MINOR | Hero problem sentence uncited | Accepted | Frame as the team's problem statement |
| 1.9 | MINOR | [4] missing volume and DOI | Accepted (Procedia Computer Science 3, 10.1016/j.procs.2010.12.125) | Add |
| 1.10 | MINOR | "tek dokunuşla düzeltir" vs ±0.5 stepper | Accepted | "birkaç dokunuşla" |

## Part 2: chef's screen (12 findings)

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 2.1 | INCONSISTENCY | Report simulates "whole amount cooked + up to 12% extra"; the screen shows İlk + batches + a "yedek (gerekirse)" cooked only if needed (reserve 11.5% of quantity at 10%); "tazeleme" means two things | Confirmed (newsvendor.simulate vs refillPlan). The reserve not being cooked would lower waste, so the report is conservative on waste | State the report's assumption; call the 12% "ek pişirme" |
| 2.2 | MISLABEL | Risk "ilk pişirilen miktar" vs card "İlk:" | Confirmed (same as 1.1) | "önerilen toplam miktar" |
| 2.3 | OVERSTATEMENT | Risk as a probability; in July demand exceeds the plan 22.7 / 12.1 / 7.7% at 20 / 10 / 5% | Accepted | "Hedeflenen"; show the realised rate per level |
| 2.4 | INCONSISTENCY | Approved/adjusted quantity frozen, not flagged when inputs change | Confirmed (finalUnits = action.units) | Flag "güncel öneri" when it differs |
| 2.5 | MISLABEL | Refill-rule sentence vs code (half a unit left, slot start −15 min, ceil 40%) | Confirmed | Describe the rule as coded |
| 2.6 | INCONSISTENCY | "etkisi henüz ölçülemedi" vs synthetic comparison in demo.json | Confirmed (same as 1.5) | "sentetik veride belirgin etki görülmedi" |
| 2.7 | MISLABEL | "tüketildi" is the corrected history; "beklenen tüketim" are demand quantiles | Confirmed (same as 1.4) | "tahmini talep (tükenen servisler düzeltilerek)", "beklenen talep" |
| 2.8 | MINOR | Demo child share 29% outside the training range 14–26%; slider cannot show it | Confirmed | Note it, like temperature |
| 2.9 | MINOR | "+25 puan" capped at 100%; "kayıtlı" shows the scaled value | Confirmed | Show the effective change; "kayıtlı" = recorded |
| 2.10 | MINOR | Klasik plan ignores the scenario | Accepted | Say so |
| 2.11 | MINOR | Temperature filter not disclosed on the screen | Accepted | One line in the plan strip |
| 2.12 | MINOR | Özgünlük 1 "ne zaman" is a fixed profile | Accepted | Tag wording |

## Part 3: "Nasıl çalışır" and README (13 findings)

| # | Class | Finding | Verdict | Change |
| --- | --- | --- | --- | --- |
| 3.1 | ERROR | "only match and themed nights" change the share; the generator also has conference lunches (+10%) | Confirmed (generator.py conference_lunch_attendance); learned effect 0.00% (8 days < min leaf 15). Text written in the audit-1 fixes | Three effects, two learned |
| 3.2 | INCONSISTENCY | "first cook" / "tazeleme" mean two things; evaluation assumes perfect-knowledge +12% extra; refill schedule not evaluated | Confirmed (same as 1.1, 2.1) | Planned amount; "ek pişirme"; state the assumption |
| 3.3 | OVERSTATEMENT | Risk presented as the actual probability on nasıl | Confirmed (realised 42.3 … 5.5%) | "Hedeflenen"; calibration note |
| 3.4 | ERROR | README lists attendance noise as hard-coded; it is in the config | Confirmed (attendance.noise_sd) | Removed |
| 3.5 | INCONSISTENCY | README "child share kept within the training range"; demo week 29% is above 26% | Confirmed (same as 2.8) | Reworded |
| 3.6 | MISLABEL | "Toplam atık" is buffet overproduction; plate waste not modelled | Confirmed | "Üretim fazlası"; README definition |
| 3.7 | MINOR | Guest crossing count included blind-week rows (predictions only) | Confirmed | Counted on open weeks |
| 3.8 | MINOR | Guest card "(MAÇ, TEMA GECESİ)" reads as the inputs | Accepted | "öğrenilen: …" |
| 3.9 | MINOR | MAE note: over which meals | Accepted | 579 meals, open part |
| 3.10 | MINOR | Three neighbours listed, model uses five | Confirmed | Said so |
| 3.11 | MINOR | Meta description "harita" | Confirmed | "liste" |
| 3.12 | MINOR | Assumptions are in two config files | Confirmed | Said so |
| 3.13 | MINOR | "Evdeki misafir" means "at home" | Confirmed | "Otelde konaklayan" |

## Round summary

| Part | Tokens | Findings | Errors |
| --- | --- | --- | --- |
| 1 | 152k | 10 | 0 (1.1 mislabel introduced by the audit-1 fix) |
| 2 | 162k | 12 | 0 |
| 3 | 166k | 13 | 2 (both in text written during the audit-1 fixes) |
| Total | about 480k | 35 | 2 |

Compared with audit 1: 43 findings, 4 errors. The audit-1 errors (explanations, new-dish claim, guest drivers,
embedding claim) did not recur; the new errors were introduced while fixing audit 1, as in the capstone's audit 8.

## Fixes applied (2026-10-24, deployed; record `2026-10-24-audit2-fixes`, models identical)

All accepted findings fixed in index.html, demo.html/demo.js, nasil.html/nasil.js and the README. Code changes:
train.py reports the July shortfall per level and counts guest crossing on open weeks only; demo.js flags an
approved amount when the recommendation has moved, shows the effective occupancy change, the recorded presence
counts, and the child-share range note; the report gains a "talebin planı aştığı yemekler" tile against the
targeted risk and states the evaluation's assumptions. Tests 8/8, parity 8 models OK, scripted browser test
without errors. Audit 3 owed under the repeat-until-clean rule (not run: the student asked to stop after audit 2).
