# Ölçülü (prototype)

> **These results come from synthetic data and are not evidence of accuracy.**
> A model trained on synthetic data learns back the assumptions of whoever generated the data.

Ölçülü forecasts how much of each dish to cook at the open buffets of all-inclusive resorts, and
suggests refill times during service with a simple rule. This repository holds the synthetic data
generator, the two models, and the static prototype website.

Live site: https://olculu.vercel.app

## Setup and running

Requires [uv](https://docs.astral.sh/uv/) and Node (Node only for the JS parity check).

```
uv sync
uv run python -m olculu.synth --seed 42                 # training season (seed 42) + test season (seed 43, shifted parameters)
uv run python -m olculu.train --name audit2-fixes       # trains, evaluates, writes experiments/<date>-<name>/ and web/data/*.json
uv run pytest                                           # leakage, time split, config sources
uv run python tests/js_parity.py experiments/<date>-<name>   # browser model == LightGBM, every quantile (predictions and SHAP)
```

`--name` sets the experiment record's folder; the default (`baseline`) would overwrite an earlier record of
the same day. `--embeddings st` uses `sentence-transformers` (`uv sync --extra embeddings`) instead of the
offline character n-gram TF-IDF. The current site uses TF-IDF; the pretrained model has not been run yet.

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
  regression (0.1/0.5/0.9) then learns the share of those guests who come to the buffet. Its inputs include
  events, weather, occupancy and guest mix, but in the synthetic generator only match nights, themed nights and
  conference lunches change that share. It learned the first two (match −4.7% vs −6% in the generator, themed
  +4.0% vs +5%) and not the conference effect (+10% in the generator; only 8 conference lunches in training,
  fewer than the trees' minimum leaf size of 15). An earlier version learned everything from counts; a sensitivity check
  showed it reacted 2–5× too weakly to excursions and à la carte bookings, depending on the size of the change,
  so the rule was made explicit. Caveat: the synthetic generator uses the same rule, so its benefit on synthetic
  data is circular, and the learned rate adds little on top of it (see Results).
- **Consumption model** (`models.fit_dish_models`): one LightGBM per quantile, target = kg per 100 predicted guests.
  The guest prediction used for training is out-of-fold (grouped by week), so the quantiles include guest-forecast error.
  Quantile crossing is fixed by sorting each row; the number of fixed rows is reported in `metrics.json`.
- **Risk setting**: the chef picks one level for all dishes. It is the targeted probability that demand exceeds
  the recommended total amount (the (1 − risk) quantile, interpolated in `models.interpolate_quantile`, rounded
  up to half a unit; `newsvendor.py` holds the rounding and the service simulation). The quantiles are not yet
  calibrated, so the realised rate differs (see Results). The evaluation assumes the whole recommended amount is
  prepared and that the kitchen can cook up to 12% of the plan extra when demand exceeds it, at the right moment
  and in the right amount; so the share of dishes that actually run out is lower than the setting. The demo's
  batch schedule (below) only splits the plan and is not part of the evaluation.
- **The models run in the browser.** `export_web.py` writes the trees as JSON; `web/js/model.js` evaluates
  them (and has an exact TreeSHAP). `tests/js_parity.py` checks every quantile model against LightGBM.
- **Explanations** (`web/js/demo.js`): counterfactual, not SHAP. Each dish sentence gives the expected guests
  and the dish's own average demand per 100 guests so far in the season (the stock-out-corrected training target,
not raw consumption); a temperature reason is added only
  when changing the temperature to a mild 27 °C moves the median by at least 5% *and* in the direction the
  catalogue expects for that dish (catalogue field, used only for this check, never as a model input). Event
  effects on guests are the forecast with the event on minus off. SHAP was used in the first version; site
  audit 1 showed that on this synthetic data it attributed noise (temperature with the wrong sign on 110 of 409
  demo dishes, child share on breakfasts), so it is not shown. No LLM.
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
1.057 vs 1.104 needed). It assumes guests who arrive after the stock-out would have eaten like the earlier ones.
A naive model (observed = demand) is trained alongside: so far there is no measurable difference.

### Train / test split

The test season is a separate generator run with a different seed and shifted parameters
(`config/synthetic.yaml: test_shift`). Its last 3 weeks are the blind test. They are excluded from all
evaluation and from the demo, and have not been opened.

Known small leak: `first_meal_share` (a dish-model input) is generated as first-meal guests divided by the
meal's actual attendance; a real system would divide by the forecast. The same-day temperature is also
used as if forecast perfectly.

## Results so far (synthetic, record `experiments/2026-10-24-audit2-fixes/`)

See `metrics.json` in the record for every number. "Waste" is buffet overproduction (prepared minus taken);
plate waste is not modelled. Stock-outs are counted two ways: the share of dishes served that ran out (after
extra cooking), and the share of meals in which at least one dish ran out.

| Test season (open part) | Waste | Demand above the plan | Dishes run out | Meals with a dish run out |
| --- | --- | --- | --- | --- |
| Simulated chef | 65.5 t | 15.8% | 7.8% | 63.7% |
| Model, 20% risk | 46.0 t | 17.0% | 7.4% | 69.4% |
| Model, 10% risk | 63.6 t | 9.1% | 3.2% | 40.6% |
| Model, 5% risk | 78.5 t | 5.5% | 1.8% | 26.8% |

The "demand above the plan" column is the realised version of the setting (uncalibrated quantiles, plus rounding
up to half units). So the setting is a real trade-off: at 20% the model wastes about 30% less than the chef but more meals see a
dish run out; at 10% waste is about the chef's and dishes run out far less often.

- Calibration is not done yet: q0.95 covered 92.7% instead of 95%, and the guest 10–90% interval covered 73%
  instead of 80%. Conformal calibration is planned for the mentoring period.
- Guest model, mean absolute error per meal: 16.9 (in-house guests × the meal's average rate), 14.0 (presence
  rule × the meal's average rate), 13.1 (presence rule + rate model). A LightGBM on occupancy only reached 17.9.
  Most of the gain is the rule, which is circular on synthetic data.
- Censoring-aware vs naive: no measurable difference yet (q0.9 coverage on dishes that often sell out 84.9% vs
  85.4%). The feedback-loop experiment is still to be done.
- New dishes: the dish-similarity input (`knn_per100`) barely moves the forecast, and the TF-IDF neighbours are
  spelling-based (Pelmeni's neighbours include Dondurma).

### What the demo lets visitors change, and what it hides

Every control in the demo's scenario panel was checked first (vary it across its range on all demo services,
confirm the forecast moves visibly and in the direction a chef would expect). Kept: occupancy (in points,
capped at 100% and capacity, the effective change shown), temperature (kept within the training range), share
of children (the slider stays within the training range, 14–26%; the demo week's own value, 29%, lies above it
and the page says so),
excursions, early departures, à la carte bookings, late-night arrivals, match night, themed night.
Not shown: nationality mix (the generator gives dishes nationality affinities, but the dish model did not learn
them: a Russian-heavy mix raised English breakfast items), conference groups (the rate model ignores them; too
few conference days), late arrivals at breakfast (too small an effect).

### Assumptions outside the config

`config/synthetic.yaml` holds the main assumptions with a source per parameter; `config/dishes.yaml` holds the
dish catalogue (take rates, portions, affinities, temperature sensitivity) under one blanket "team assumption".
Some smaller ones are hard-coded in
`synth/generator.py`: the shapes of the summer and temperature curves, noise levels on arrivals, departures
and arrival profiles, the first-meal fractions, the attendance floor, the occupancy clip, the late
wave's timing, and the simulated chef's information (the previous same meal's attendance with 5% noise, and a
noisy first guess for new dishes). Moving them into the config is planned.

## Website design choice

Considered: (1) a single long landing page with an embedded demo, (2) a slide-like scroller, (3) a short
landing page plus a separate tablet-first "chef's screen". Chose (3): jurors get the story in a few minutes
on the landing page, and the chef's screen can be designed purely for a kitchen tablet (large touch targets,
one question: what to prepare now and how much) without fighting the marketing layout. The visual style
follows the team's pitch deck (not the rejected mock-up).

The product name lives in `web/js/site-config.js` (site) and is a one-line change.

## Audits

The site, README and code claims are audited by independent agents until an audit finds nothing (`audit/`).

## Not done yet (mentoring period)

Conformal calibration (CQR, per meal and day type), the five-way model comparison harness, the hierarchical
meal-total model, the feedback-loop experiment and chart, the embedding vs category vs TF-IDF comparison on new
dishes, the sentence-transformers run, a learned arrival profile, and opening the blind test once.
