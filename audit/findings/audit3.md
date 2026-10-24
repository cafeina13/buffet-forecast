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
