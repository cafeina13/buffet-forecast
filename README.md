# Ölçülü (prototype)

> **These results come from synthetic data and are not evidence of accuracy.**
> A model trained on synthetic data learns back the assumptions of whoever generated the data.

Ölçülü forecasts how much of each dish to cook, and when, at the open buffets of all-inclusive
resorts. This repository holds the synthetic data generator, the two models, and the static
prototype website.

Live site: https://olculu.vercel.app

## Setup and running

Requires [uv](https://docs.astral.sh/uv/) and Node (Node only for the JS parity check).

```
uv sync
uv run python -m olculu.synth --seed 42     # training season (seed 42) + test season (seed 43, shifted parameters)
uv run python -m olculu.train               # trains, evaluates, writes experiments/<date>-baseline/ and web/data/*.json
uv run pytest                               # leakage, time split, config sources
uv run python tests/js_parity.py            # browser model == LightGBM (predictions and SHAP)
```

`--embeddings st` uses `sentence-transformers` (`uv sync --extra embeddings`) instead of the
offline character n-gram TF-IDF fallback. The current site uses TF-IDF.

## Deploying

The site is plain static files in `web/` (no build step). One command:

```
cd web && vercel deploy --prod
```

`web/.vercelignore` keeps local files (`.env*`, `_*.html` preview helpers) out of the deployment.
No keys or account details are stored in the code.

## Architecture

```
dish name + ingredients -> embedding (ready-made / TF-IDF) --+
                                                             +-> consumption model (LightGBM quantiles 0.1/0.5/0.8/0.9/0.95)
occupancy, flights, excursions, events -> guest model -------+        -> risk setting -> quantity, refill times, Turkish reason
```

- **Guest model** = a presence rule × a learned rate. The rule (`features.present_guests`, not learned) counts
  guests physically in the hotel at the meal from front-office records: excursions miss lunch, à la carte
  bookings and late-night arrivals miss dinner, early departures miss lunch or breakfast. LightGBM quantile
  regression (0.1/0.5/0.9) then learns the share of those guests who come to the buffet (events, weather,
  occupancy, guest mix). An earlier version learned everything from counts; a sensitivity check showed it
  reacted 4–5× too weakly to excursions and à la carte bookings, so the rule was made explicit.
  Caveat: the synthetic generator uses the same rule, so its benefit on synthetic data is partly circular.
- **Consumption model** (`models.fit_dish_models`): one LightGBM per quantile, target = kg per 100 predicted guests.
  The guest prediction used for training is out-of-fold (grouped by week), so the quantiles include guest-forecast error.
  Quantile crossing is fixed by sorting each row; the number of fixed rows is reported in `metrics.json`.
- **Risk setting** (`newsvendor.py`): the chef picks the risk that *this dish* runs out (per dish, not the whole
  buffet); the quantity is the matching (1 − risk) quantile, interpolated between the trained quantiles.
- **The models run in the browser.** `export_web.py` writes the trees as JSON; `web/js/model.js` evaluates
  them and computes exact TreeSHAP. `tests/js_parity.py` checks both against LightGBM.
- **Explanations**: SHAP values of the median consumption model; the two strongest context features
  (temperature, events, menu competition, guest mix) are turned into one Turkish sentence by template. No LLM.
- **Refills** (`web/js/demo.js`): rule-based. About 40% is cooked first; the rest follows in batches of about
  a fifth, each scheduled 15 minutes before the tray is expected to be half empty, given the hourly arrival profile.

### How stock-outs are handled (censored demand)

For a service that ran out at time t, observed consumption only covers the guests who had arrived by t.
The censoring-aware target scales it by 1 / F(t), where F(t) is the share of the service's guests that had
arrived by t (from the hourly flow), capped at ×2 (`features.censoring_corrected`). This is the simpler of the
two options in the brief and is easy to explain to a chef. It assumes guests who arrive after the stock-out
would have eaten like the earlier ones. A naive model (observed = demand) is trained alongside for comparison.

### Train / test split

The test season is a separate generator run with a different seed and shifted parameters
(`config/synthetic.yaml: test_shift`). Its last 3 weeks are the blind test. They are excluded from all
evaluation and from the demo, and have not been opened.

## Results so far (synthetic, record `experiments/2026-10-24-guest-rule/`)

See `metrics.json` in the record for every number. Summary:

- Waste vs stock-outs over the test season: at 20% risk per dish, the model wasted about 30% less than the
  simulated chef while running out slightly less often (7.4% vs 7.8% of services). At 5% risk it wastes more than the chef.
- Calibration is not done yet: q0.95 covered 92.7% instead of 95%, and the guest 10–90% interval covered 73%
  instead of 80%. Conformal calibration is planned for the mentoring period.
- Guest model: mean absolute error 13.1 guests per meal (presence rule + rate model) vs 17.9 with occupancy only
  (partly circular, see above). The earlier all-learned version (record `2026-10-24-baseline`) reached 16.5.
- Censoring-aware vs naive: no measurable difference yet in this generator (stock-outs are mild). The
  feedback-loop experiment is still to be done.

### What the demo lets visitors change, and what it hides

Every control in the demo's scenario panel was checked first (vary it across its range on all demo services,
confirm the forecast moves visibly and in the direction a chef would expect). Kept: occupancy, temperature,
share of children, excursions, early departures, à la carte bookings, late-night arrivals, match night, themed night.
Not shown: nationality mix (the dish model did not learn nationality-dish affinities from this data; a
Russian-heavy mix raised English breakfast items), conference groups (the rate model ignores them; too few
conference days), late arrivals at breakfast (too small an effect). The same rule applies to the Turkish
explanations: weekday and nationality stay in the models but are never used in a sentence, because the
generator gives them no effect and any SHAP credit they get is noise.

## Website design choice

Considered: (1) a single long landing page with an embedded demo, (2) a slide-like scroller, (3) a short
landing page plus a separate tablet-first "chef's screen". Chose (3): jurors get the story in a few minutes
on the landing page, and the chef's screen can be designed purely for a kitchen tablet (large touch targets,
one question: what to prepare now and how much) without fighting the marketing layout. The visual style
follows the team's pitch deck (not the rejected mock-up).

The product name lives in `web/js/site-config.js` (site) and is a one-line change.

## Not done yet (mentoring period)

Conformal calibration (CQR, per meal and day type), the five-way model comparison harness, the hierarchical
meal-total model, the feedback-loop experiment and chart, the embedding vs category vs TF-IDF comparison on new
dishes, the sentence-transformers run, and opening the blind test once.
