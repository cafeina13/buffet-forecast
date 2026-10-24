# Audit 1 (2026-10-24): partitioned, three agents one at a time, no hints

Input frozen at `../exports/2026-10-24-audit1/`; brief `../brief.md`.
Fixes are applied only after all three parts report. DRAFT while parts run.
Claude's checks: `../scripts/audit1/v_claude_part1.py`.

## Part 1: home page and references (148k tokens, 13 findings)

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 1.1 | ERROR | #yapay-zeka: "hazır bir metin modeli" / "HAZIR METİN MODELİ" | Confirmed: published record uses `embeddings: tfidf` (character n-gram TF-IDF fitted on the catalogue); sentence-transformers never run | Say TF-IDF; ready-made model planned |
| 1.2 | MISLABEL | Şefin ekranı card example "Bugün 4 tepsi; doluluk %8 düştü, hava 34 °C." | Confirmed: occupancy is not a dish-model feature; templates state levels, never changes | Quote a sentence the demo produces |
| 1.3 | OVERSTATEMENT | "literatüre dayanan varsayımlarla üretilmiş sentetik veri" | Confirmed: 56 of 64 parameters `assumption`, 7 `design`, 1 `sourced` (CO₂ 1.9). Sub-point rejected: meal_scale "opposite to the cited direction" (realised kg/guest dinner 0.90 > breakfast 0.25; breakfast scale compensates a shorter catalogue, as the config says) | "Çoğu ekip varsayımı; birkaç yönü literatürden" |
| 1.4 | OVERSTATEMENT | Hero, meta, hero-problem, Çözüm card 2: timing "tahmin" | Confirmed: refill times from `refillPlan()` rule over the generator's assumed arrival profile; only quantity is modelled | Quantity predicted; refill times by rule over an assumed arrival curve |
| 1.5 | OVERSTATEMENT | Misafir modeli "maç, tema gecesi ve havaya göre öğrenir" | Confirmed: generator attendance has no temperature term (synth/generator.py) | Drop weather from the guest card, or say it has no effect in synthetic data |
| 1.6 | MISLABEL | Tüketim modeli "Tükenme kayıtları ‘en az bu kadar’ olarak işlenir" | Confirmed: point scaling ×min(1/F(t),2), trained as exact; median factor 1.057 vs true 1.104, never above true; q0.9 coverage on sold-out-prone rows 84.9% aware vs 85.4% naive. Same wording on the demo and nasıl pages | Describe the scaling; state no measured gain yet |
| 1.7 | OVERSTATEMENT | Çözüm card 3 "model her gün kendini düzeltir" | Confirmed: trained once; demo actions only in localStorage | Future tense; loop not built |
| 1.8 | MINOR | "her yemeğin bitme riskini şef kendisi seçer" | Confirmed: one level for all dishes; level = P(demand > recommended), not stock-out rate (7.4% / 3.2% / 1.8% at 20/10/5%) | Reword with the definition |
| 1.9 | MINOR | "gerekçe (SHAP)" explains the median model, not the shown quantile | Confirmed (`shap(D.models[1])`) | Say so |
| 1.10 | MINOR | "Saatlik tazeleme": slots are 15 min | Confirmed | "Servis içi tazeleme" |
| 1.11 | MINOR | "Ek donanım gerekmez" vs "tartılır" | Accepted | Conditional wording |
| 1.12 | MINOR | Hero problem sentence stated as fact, unsourced | Accepted | Soften |
| 1.13 | MINOR | [1] subtitle missing; [1] not cited in body; [2] body wording beyond the title; contact placeholder | Accepted except the contact placeholder (the student's decision) | Fix the three reference points |

## Part 2: chef's screen (168k tokens, 15 findings)

Claude's checks: `../scripts/audit1/v_claude_part2.js`.

| # | Class | Finding | Verdict | Change (pending) |
| --- | --- | --- | --- | --- |
| 2.1 | OVERSTATEMENT | Özgünlük 2 strip and "Bitti" status: "model … ‘en az bu kadar yenirdi’ diye öğrenir ve … daha az önermez" | Confirmed (same as 1.6): taps only go to localStorage; correction is point scaling; no measured gain | Present as design for the pilot, effect not yet measured |
| 2.2 | ERROR | New-dish sentence "tahmin, benzer yemeklerden (…) geliyor" | Confirmed: Tavuk kanat 15 Jul lunch, knn_per100 from 1 to 12 moves the median 2.99 -> 3.23 kg/100; SHAP -0.001 | Drop the sentence; say no history, forecast near the general level |
| 2.3 | ERROR | Temperature clauses contradict the generator (16 of 90 shown clauses) | Confirmed and broader: of 409 demo dish rows, temperature SHAP has the wrong sign vs the generator on 110 and credits 100 dishes the generator gives no temperature effect | Rethink dish explanations (see 2.4) |
| 2.4 | OVERSTATEMENT | Two whitelisted reasons presented as the reasons; history (hist_per100) is the top driver on 370 of 409 cards; cited reasons median 1.9% of the prediction; event clauses per dish; first-meal and menu-competition clauses are artefacts of a cross-meal baseline | Accepted (code: whitelist, top-2 by abs SHAP, no size threshold, base over all meals) | Rethink: lead with history; only effects whose direction matches known behaviour and clears a size threshold; relative to the same meal |
| 2.5 | ERROR | Guest driver "çocuklu misafir +3–4 kişi" on breakfasts | Confirmed: generator attendance has no child, temperature or occupancy term (synth/generator.py); contradicts the comment above GUEST_EXPLAINED | Guest drivers: events only |
| 2.6 | INCONSISTENCY | "maç akşamı −17 kişi" vs toggle | Confirmed: 15 Jul dinner SHAP -17.5 guests, counterfactual on-off -27.6 | Show the counterfactual difference |
| 2.7 | MISLABEL | "Tükenen servis oranı" is a share of dish-servings | Confirmed (simulate(): mean over rows) | "Tükenen yemek oranı (yemek × servis)" |
| 2.8 | MISLABEL | 5% tile "Ek üretim" shows extra waste | Confirmed (-avoided_kg) | "Ek atık" |
| 2.9 | INCONSISTENCY | Chosen risk vs realised run-out rate; 12% refill allowance unstated; under-calibration | Accepted (q0.9 covers 87.6%, q0.95 92.7%) | One explanatory line in the report; mention calibration |
| 2.10 | OVERSTATEMENT | Arrival curve labelled "(tahmin)", "saat saat"; live clock includes the current slot | Confirmed (export uses generator.arrival_profile; same with 1.4) | "Varsayılan geliş profili"; 15-min; end-of-slot |
| 2.11 | OVERSTATEMENT | "Tasarruf" is only the cost of avoided waste; at 20% stock-outs rise (10.3% vs 8.2%) | Confirmed | "Önlenen atığın maliyeti"; state the stock-out trade-off |
| 2.12 | MINOR | "YENİ" badge regardless of history | Confirmed (badge from catalogue freq) | Badge only without history |
| 2.13 | MINOR | Sliders outside training support; occupancy scales presence uncapped | Accepted | Clamp ranges; cap in_house; points |
| 2.14 | MINOR | Breakdown rounding (748 × %78 = 580) | Accepted | One decimal or note rounding |
| 2.15 | MINOR | "Çocuklu misafir payı" vs share of children; "Bugün" on every day | Accepted | "Çocuk payı"; drop "Bugün" |

Noted for the other parts (not passed to their auditors): the monthly report compares at a fixed risk, and in July at 20% the model's stock-out rate (10.3%) is above the chef's (8.2%).
