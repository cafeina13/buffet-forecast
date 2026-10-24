# TrayCast (prototype)

> **These results come from semi-synthetic data and are not evidence of accuracy.**
> Guest flow, weather and match nights are real; dish demand is simulated, and a model trained on simulated
> demand learns back the assumptions of whoever generated it.

TrayCast (formerly the working name Ölçülü) forecasts how much of each dish to cook at the open buffets of
all-inclusive resorts, and suggests refill times during service with a simple rule. This repository holds the
data generator, the two models, and the static prototype website.

Live site: https://traycast.vercel.app

## Setup and running

Requires [uv](https://docs.astral.sh/uv/) and Node (Node only for the JS parity check).

```
uv sync
uv run python -m olculu.synth.fetch                     # downloads the real inputs into data/raw/ (see below)
uv run python -m olculu.synth --seed 42 --grounded      # semi-synthetic training season (2016) + test season (2024 calendar)
uv run python -m olculu.train --name grounded-final     # trains, evaluates, writes experiments/<date>-<name>/ and web/data/*.json
uv run pytest                                           # see "Tests" below
uv run python tests/js_parity.py experiments/<date>-<name>   # browser model == LightGBM, every quantile (predictions and SHAP)
```

Without `--grounded` the generator builds the earlier fully synthetic seasons (2026). `--name` sets the experiment
record's folder; the default (`baseline`) would overwrite an earlier record of the same day. `--embeddings st`
uses `sentence-transformers` (`uv sync --extra embeddings`) instead of the offline character n-gram TF-IDF. The
site uses TF-IDF; the pretrained model has not been run yet.

## Semi-synthetic data

What is real (`config/grounding.yaml`, `synth/grounding.py`; every entry has a source):

| Input | Source | Licence |
| --- | --- | --- |
| Guests staying, arriving and departing each day; share of children | Hotel Booking Demand, Resort Hotel H1, non-cancelled bookings (Antonio, de Almeida & Nunes 2019, *Data in Brief* 22:41–49), scaled so the busiest night is 96% of our 420-room resort | CC BY 4.0 |
| Daily maximum temperature, Antalya (36.90 N, 30.71 E) | Open-Meteo historical weather API (ERA5) | free use |
| Match nights | Evening matches (kick-off 18:00 CEST = 19:00 Antalya or later) of Türkiye, Germany or England at Euro 2016 and Euro 2024 (openfootball) | CC0 1.0 |

Still simulated: dish demand, excursions and à la carte bookings, the late-arrival and early-departure shares of
the real arrivals and departures, the nationality mix, themed nights and conferences.

Seasons: training = 2016 (bookings, weather and Euro 2016 of the same year); test = the 2024 calendar (2017
booking pattern placed on 2024 by month and day, 2024 weather, Euro 2024). The booking data ends in August 2017,
so the test season runs 1 April–31 August 2024; its last 3 weeks are the blind test.

Limits: the guests come from a resort in Portugal, not Antalya (e.g. child share 1–16%, probably lower than in
Antalya family resorts; average stay about 4.7 nights). Summer 2024 was hotter than summer 2016 (June–July average
high 37.3 °C vs 33.1 °C); 29 of the 153 test days were hotter than the hottest training day (38.7 °C, test maximum
43.2 °C), so the model meets temperatures it never saw. The trees treat them like the hottest day they know; the
demo's temperature slider is clamped to the training range.

## Deploying

The site is plain static files in `web/` (no build step). One command:

```
cd web && vercel deploy --prod
```

`web/.vercelignore` keeps local files (`.env*`, `_*.html` preview helpers) out of the deployment.
No keys or account details are stored in the code.

## Architecture

```
dish name + ingredients -> TF-IDF dish vector -----------------+
                                                               +-> consumption model (LightGBM quantiles 0.1/0.5/0.8/0.9/0.95)
front-office counts -> presence rule x guest rate model -------+        -> risk setting -> quantity; refill rule; Turkish reason
```

- **Guest model** = a presence rule × a learned rate. The rule (`features.present_guests`, not learned) counts
  guests physically in the hotel at the meal from front-office records: excursions miss lunch, à la carte
  bookings and late-night arrivals miss dinner, early departures miss lunch or breakfast. LightGBM quantile
  regression (0.1/0.5/0.9, minimum leaf 8 days) then learns the share of those guests who come to the buffet.
  Its inputs include events, weather, occupancy and guest mix, but in the generator only match nights, themed
  nights and conference lunches change that share. It learned all three, weakly: match −1.8% (−6% in the
  generator), themed +3.6% (+5%), conference +4.8% (+10%); the training season has only 10 real match nights and
  13 conference lunches. With the earlier minimum leaf of 15 days the 10 match nights could not be learned at all
  (effect 0.0%). These figures are in `metrics.json` under `method_facts`. An earlier version learned everything
  from counts; a sensitivity check showed it reacted 2–5× too weakly to excursions and à la carte bookings,
  depending on the size of the change, so the rule was made explicit. Caveat: attendance is generated with the
  same rule, so its benefit is circular, and the learned rate adds little on top of it (see Results).
- **Consumption model** (`models.fit_dish_models`): one LightGBM per quantile, target = kg per 100 predicted guests.
  The guest prediction used for training is out-of-fold (grouped by week), so the quantiles include guest-forecast error.
  Quantile crossing is fixed by sorting each row; the number of fixed rows is reported in `metrics.json`.
- **Risk setting**: the chef picks one level for all dishes with a slider from 5% to 30% (1-point steps; the
  season report is precomputed for every step). A marker shows where the simulated chef's classic plan sits: the
  setting at which the model's demand-above-plan rate equals the chef's (about 21%), interpolated from the
  season report. The level is the targeted probability that demand exceeds
  the recommended total amount (the (1 − risk) quantile, interpolated in `models.interpolate_quantile`, rounded
  up to half a unit; `newsvendor.py` holds the rounding and the service simulation). The realised rate differs
  from the setting through two effects in opposite directions: the quantiles are not yet calibrated (demand
  exceeds the unrounded quantile more often than the setting, 22.3% at 20%), and rounding up to half units makes
  it exceed less often (16.9% at 20%). The evaluation assumes the whole recommended amount is prepared and that
  the kitchen can cook up to 12% of the plan extra when demand exceeds it, at the right moment and in whole units;
  so the share of dishes that actually run out is lower than the setting. The demo's batch schedule (below) only
  splits the plan and is not part of the evaluation.
- **The models run in the browser.** `export_web.py` writes the trees as JSON; `web/js/model.js` evaluates
  them (and has an exact TreeSHAP). `tests/js_parity.py` checks every quantile model against LightGBM.
- **Explanations** (`web/js/demo.js`): counterfactual, not SHAP. Each dish sentence gives the expected guests
  and the dish's own average demand so far in the season (stock-out-corrected consumption per 100 actual guests,
  not raw consumption; the training target divides by predicted guests instead); a temperature reason is added only
  when the day is at least 3 °C from 27 °C and changing the temperature to 27 °C moves the median by at least 5%
  *and* in the direction the catalogue expects for that dish (catalogue field, a team assumption that also drives
  the simulated demand; used only for this check, never as a model input). Event effects on guests are the forecast
  with the event on minus off. SHAP was used in the first version; site audit 1 showed that on the fully synthetic
  data it attributed noise (temperature with the wrong sign on 110 of 409 demo dishes, child share on breakfasts),
  so it is not shown. No LLM.
- **Refill schedule** (`web/js/demo.js`): rule-based, splitting the recommended amount. About 40% first (rounded
  up to whole units); the rest in batches of about a fifth, each scheduled 15 minutes before the start of the
  15-minute slot in which the median consumption is expected to bring the buffet down to half a unit (never
  before service start). Units that a median service would not need are shown as a reserve "if needed". The
  arrival profile is assumed (the generator's curve, with a match variant; not learned).

### How stock-outs are handled (censored demand)

For a service that ran out at time t, observed consumption only covers the guests who had arrived by t.
The target scales it by 1 / F(t), where F(t) is the share of the service's guests that had arrived by t
(from the hourly flow), capped at ×2 (`features.censoring_corrected`). This is a point estimate used as if
exact, not a lower-bound (censored) loss. On the training season it stays below true demand (median factor
1.060 vs 1.117 needed; `method_facts`). It assumes guests who arrive after the stock-out would have eaten like
the earlier ones. A naive model (observed = demand) is trained alongside (see Results).

### Train / test split

Training and test seasons are separate generator runs (different seeds, shifted parameters in
`config/synthetic.yaml: test_shift`, and different real years). The test season's last 3 weeks are the blind
test. They are excluded from all evaluation and from the demo, and have not been opened.

Known small leak: `first_meal_share` (a dish-model input) is generated as first-meal guests divided by the
meal's actual attendance; a real system would divide by the forecast. The same-day temperature is also
used as if forecast perfectly.

## Results so far (semi-synthetic, record `experiments/2026-10-24-grounded-final/`)

The evaluation numbers below are in `metrics.json` in the record (`risk_curve`, `chef`, `season_report` with a
per-month breakdown, `guest`, `censoring`, `method_facts`). A few method facts (the 2–5× sensitivity of an
earlier model, the SHAP sign count of the first version) come from audit scripts under `audit/scripts/`.
"Waste" is buffet overproduction (prepared minus taken); plate waste is not modelled. Stock-outs are counted two
ways: the share of dishes served that ran out (after extra cooking), and the share of meals in which at least one
dish ran out. Test season, open part: 1 April–10 August 2024, 396 meals.

| Test season (open part) | Waste | Demand above the plan | Dishes run out | Meals with a dish run out |
| --- | --- | --- | --- | --- |
| Simulated chef | 40.4 t | 17.5% | 9.5% | 71.2% |
| Model, 20% risk | 29.8 t | 16.9% | 6.9% | 67.7% |
| Model, 10% risk | 40.0 t | 9.9% | 3.6% | 48.5% |
| Model, 5% risk | 49.3 t | 5.5% | 1.9% | 29.8% |

At 20% the model wastes about 26% less than the chef and runs out less often on both measures; at 10% its waste is
about the chef's and dishes run out far less often. The result varies by month: at 10% the model's waste ranges
from 9.3% below the chef's (August) to 5.0% above it (June); at 10% over the season it wastes 0.4 t less but costs
about 160 thousand TL more, because its extra waste falls on dearer dishes.

- Calibration is not done yet: q0.95 covered 92.1% instead of 95%, and the guest 10–90% interval covered 69%
  instead of 80%. Conformal calibration is planned for the mentoring period.
- Guest model, mean absolute error per meal: 20.2 (in-house guests × the meal's average rate), 15.2 (presence
  rule × the meal's average rate), 14.1 (presence rule + rate model). A LightGBM on meal, in-house guests and
  occupancy only reached 21.2. Most of the gain is the rule, which is circular here.
- Censoring-aware vs naive: on dishes that often sell out, q0.9 covers 86.8% with the correction vs 85.1% without
  (363 dish-servings). Small, but in the expected direction; the fully synthetic version showed no difference.
  The feedback-loop experiment is still to be done.
- Child share: with real, day-to-day varying child shares the dish model learns the child effect in the right
  direction. Probe (`audit/scripts/audit3/child_probe.py`) on an earlier semi-synthetic record: +6.4% learned vs
  +5.2% in the generator, wrong sign on 1% of dishes; on the fully synthetic data, where the child share was a
  smooth seasonal curve, the sign was wrong on about a fifth of dishes.
- New dishes: the dish-similarity input (`knn_per100`) barely moves the forecast, and the TF-IDF neighbours are
  spelling-based (Pelmeni's neighbours include Dondurma).

### What the demo lets visitors change, and what it hides

Every control in the demo's scenario panel was checked first (vary it across its range on all demo services,
confirm the forecast moves visibly and in the direction a chef would expect). Kept: occupancy (in points,
capped at 100% and capacity, the effective change shown), temperature (kept within the training range, the
effective change shown), share of children (slider within the training range), excursions, early departures,
à la carte bookings, late-night arrivals, match night, themed night. Not shown: nationality mix (still simulated;
the dish model did not learn the simulated nationality affinities), conference groups (learned only weakly; few
conference days), late arrivals at breakfast (too small an effect).

### Assumptions outside the config

`config/synthetic.yaml` holds the main assumptions with a source per parameter; `config/dishes.yaml` holds the
dish catalogue (take rates, portions, affinities, temperature sensitivity) under one blanket "team assumption";
`config/grounding.yaml` holds the real inputs. Some smaller assumptions are hard-coded in `synth/generator.py`,
among others: the shapes of the summer and temperature curves (used only in the fully synthetic mode), noise
levels on arrivals, departures and arrival profiles, the first-meal fractions and their 0.5 cap, the reference
levels that scale the child and first-meal effects (/0.2, /0.1), the 0.5-portion floor on mean demand, the
attendance floor, the occupancy clip, the late wave's timing, and the simulated chef's information (the previous
same meal's attendance with 5% noise, the last 4 observed, i.e. censored, per-guest consumptions, and a noisy
first guess for new dishes). Moving them into the config is planned.

## Tests

`uv run pytest` checks that true demand is not a feature and that changing it does not change any feature, that
history features use only earlier days, that sold-out services are scaled up and not down, that the test run
uses shifted parameters, the blind-cutoff arithmetic, that every config parameter has a source, and (when the
real inputs have been fetched) that the Euro fixtures are read as documented and the bookings are scaled to the
configured peak. It does not test that the evaluation and the demo exclude the blind weeks (checked by reading
`train.py`), and it cannot detect the documented `first_meal_share` leak.

## Website design choice

Considered: (1) a single long landing page with an embedded demo, (2) a slide-like scroller, (3) a short
landing page plus a separate tablet-first "chef's screen". Chose (3): jurors get the story in a few minutes
on the landing page, and the chef's screen can be designed purely for a kitchen tablet (large touch targets,
one question: what to prepare now and how much) without fighting the marketing layout. The visual style
follows the team's pitch deck (not the rejected mock-up).

The product name is set in `web/js/site-config.js`, but the HTML pages also carry it in their `<title>` and as
fallback text for visitors without JavaScript, so a rename touches those files too.

## Audits

The site, README and code claims are audited by independent agents until an audit finds nothing (`audit/`).
Audits 1–3 covered the fully synthetic version; this semi-synthetic version has not been audited yet.

## Not done yet (mentoring period)

Conformal calibration (CQR, per meal and day type), the five-way model comparison harness, the hierarchical
meal-total model, the feedback-loop experiment and chart, the embedding vs category vs TF-IDF comparison on new
dishes, the sentence-transformers run, a learned arrival profile, real nationality mix (e.g. Antalya monthly
statistics), and opening the blind test once.
