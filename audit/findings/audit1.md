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

Noted for the other parts (not passed to their auditors): the monthly report compares at a fixed risk, and in July at 20% the model's stock-out rate (10.3%) is above the chef's (8.2%).
