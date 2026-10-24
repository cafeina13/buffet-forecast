# Audit 3 (2026-10-25): partitioned, three agents one at a time, no hints

Input frozen at `../exports/2026-10-25-audit3/`; brief `../brief.md`.
Fixes are applied only after all three parts report. DRAFT while parts run.

## Part 1: home page and references (5 findings, 0 errors)

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 1.1 | MISLABEL | "aynı risk ayarında" (card 4, Başarı ölçütü): the chef has no risk setting; the model at the chosen setting is compared with the chef's fixed plan | Confirmed (generator chef plan = history × 1.25–1.4; only the model moves) | Reword: chosen setting vs the chef's fixed plan; rates shown side by side |
| 1.2 | MINOR | "(prototipte bu ölçüm adımı simüle edilmedi)": leftovers are simulated (±3% scale noise) but unused | Confirmed (leftover_kg written, never read) | "simüle edildi ama kullanılmıyor; tüketimin hatasız bilindiği varsayıldı" |
| 1.3 | MINOR | "mutfak beklentisi" is the catalogue's temperature sensitivity, the same assumption that generated the data | Confirmed (heat_direction from dishes.yaml) | "katalogdaki (ekip varsayımı) yön"; note the circularity |
| 1.4 | MINOR | Literature attribution incomplete (temperature from [3], which gives no direction; no reference number) | Confirmed | Add [1]/[3] and say what each supports |
| 1.5 | MINOR | Başarı ölçütü does not say which stock-out rate is held equal | Accepted | Name it (tükenen yemek oranı) |

## Part 2: chef's screen (11 findings, 0 wrong numbers)

Claude's check: `../scripts/audit3/v_claude_part2.py`.

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 2.1 | OVERSTATEMENT | "gerçekleşen oran biraz daha yüksek" holds for July only | Confirmed: shortfall at 10% by month 9.0 / 9.6 / 10.6 / 12.1 / 7.6 / 6.6 / 6.8% (Apr–Oct); season 9.1% | Say it varies by month around the setting; give July and season |
| 2.2 | OVERSTATEMENT | July is the model's best month; shown as "the" monthly report | Confirmed: waste vs chef at 10% Apr +11.2, May −1.9, Jun −3.2, Jul −11.8, Aug −6.9, Sep +1.3, Oct −1.4%; at 20% −18.5 … −39.0% (July). July chosen because the demo week falls in it, but the effect is cherry-picking | Season totals or a month selector; say July is the most favourable |
| 2.3 | MISLABEL | "yukarıdaki ilk kutu" points to the wrong tile | Confirmed | Name the tile |
| 2.4 | INCONSISTENCY | Temperature slider shows "+8 °C" while the clamp applies +1.5 °C | Confirmed | Show the effective change |
| 2.5 | INCONSISTENCY | Child-share slider offered though learned unreliably (wrong sign on 22% of rows); demo value 29% above the 14–26% range | Accepted | Hide it like nationality, or label it unreliable |
| 2.6 | MINOR | "gerektiği kadar": extra cooking is in whole units | Confirmed | "tam birim olarak" |
| 2.7 | MINOR | Plan note omits the ≥3 °C-from-27 condition | Confirmed | Add it |
| 2.8 | MINOR | Recommendation can sit above its own "usual" range (round up vs round nearest) | Confirmed | Same rounding |
| 2.9 | MINOR | "Ortalama bir talepte" is the median | Confirmed | "tipik (medyan)" |
| 2.10 | MINOR | Cost to the lira from rough estimates | Accepted | Round (≈ … bin TL) |
| 2.11 | MINOR | "Talebin planı aştığı yemekler" counts dish-servings | Confirmed | "yemek servisleri" |

## Part 3: "Nasıl çalışır" and README (12 findings, 1 error)

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 3.1 | ERROR | Realised rate blamed on calibration only; rounding up to half units pulls it the other way | Confirmed: q0.8 covers 77.6% (exceedance 22.4% unrounded) vs 17.0% after rounding up; README bullet contradicts README Results | Name both causes and their directions |
| 3.2 | MISLABEL | "(tablonun ilk sütunu)" is the second data column | Confirmed (nasil.js column order) | Name the column |
| 3.3 | MINOR | Extra cooking "gerektiği kadar" is whole units | Confirmed (same as 2.6) | "tam birimler halinde" |
| 3.4 | MINOR | "Occupancy only" model also uses meal and in-house count | Confirmed (GUEST_FEATURES_OCC_ONLY) | "öğün ve doluluk" |
| 3.5 | MINOR | Presence rule text omits departures | Confirmed | Add them |
| 3.6 | MINOR | README calls hist_per100 the training target (different denominator) | Confirmed | "per 100 actual guests" |
| 3.7 | MINOR | Child slider range 15–26%, not 14–26% | Confirmed | Moot if the slider is removed (2.5) |
| 3.8 | MINOR | "every number in metrics.json": several README facts come from audit scripts | Confirmed | Reword, point to audit/ |
| 3.9 | MINOR | Test description broader than the tests | Accepted | Describe narrowly |
| 3.10 | MINOR | Hard-coded assumption list incomplete | Confirmed | Add, "among others" |
| 3.11 | MINOR | Rename is not a one-line change (HTML titles, fallbacks) | Confirmed (eab19e2) | Say so |
| 3.12 | MINOR | Inconsistent decimals ("46 t" next to "65,5 t") | Confirmed | One decimal |

Checked by the auditor: the chef comparison barely changes without the circular presence rule (46.2 t / 7.5% / 67.9% at 20% with in-house × meal rate), so the headline does not rest on the circularity.

## Round summary

| Part | Tokens | Findings | Errors |
| --- | --- | --- | --- |
| 1 | 137k | 5 | 0 |
| 2 | 183k | 11 | 0 |
| 3 | ~170k | 12 | 1 |
| Total | about 490k | 28 | 1 |

Audits 1 / 2 / 3: 43 / 35 / 28 findings, 4 / 2 / 1 errors.

## Fixes applied (2026-10-25, deployed; record `2026-10-24-audit3-fixes`, models identical)

- 2.2 (student's choice a): the demo report is now a season report over the open test season with a
  per-month table at the chosen setting. It shows that at 10% the model saves 1.9 t but costs ≈ 60 bin TL more
  (extra waste on dearer dishes); the TL tile is signed.
- 2.5 (student's decision): the child slider stays. Diagnosis (`/tmp` probe, numbers in README): the generator's
  child share has almost no day-to-day variation (residual sd 0.0005, corr with temperature 0.74), so the effect is
  unidentifiable; with real bookings (semi-synthetic branch, residual sd 0.011) the model learns +6.4% vs +5.2% true,
  wrong sign on 1% of dishes. The page now says so next to the slider.
- 3.1, 3.8: train.py writes `method_facts` (censoring factors, learned event effects, conference count,
  unrounded exceedance); both causes of the realised-vs-setting gap are named on the demo, nasıl and README.
- All other accepted findings fixed as listed. Tests 8/8, parity 8 models, scripted browser test without errors.
