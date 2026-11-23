# WeatherNext 2 with WeatherNext 3 features: status at 1°

Status as of 2026-11-22. Code: github.com/xincjin-NOAA/wn2_gdas, branch `develop`. Full details and every run are in `docs/project_report.md`.

## Summary

We fine-tuned Google DeepMind's WeatherNext 2 (WN2) on NOAA GDAS. We then added the main new features of WeatherNext 3 (WN3) to it one at a time, each with a pass/fail test set in advance. This report uses the 1° WN2 Mini model, where each experiment takes a few GPU hours.

- **Fine-tuning on GDAS works.** 2 m temperature error (CRPS) is 45% lower at 1 day and 21% lower at 10 days than the released WN2 model.
- **The station head is the strongest result.** At stations never used in training, it cuts 2 m temperature CRPS by 35–41% out to 48 h. It also removes the forecast's cold bias and fixes most of its under-spread. WN3 reports up to 30% for its station head.
- **Hourly output works.** The model now predicts the five hours between its 6-hourly steps. In the first step it beats linear interpolation by 32% for 500 hPa height and 15% for 2 m temperature.
- **Hourly initialisation works without retraining.** Starting from an hourly GDAS forecast state instead of the last analysis cuts 6 h precipitation CRPS by 6–15% at 6–24 h, and the run is better on every test start.
- **Satellite input passes its test, but isn't proven yet.** Adding GOES/Meteosat/Himawari imagery improves upper-air humidity by about 25% and most other fields by 1–5% after 1000 fine-tune steps, compared with a model trained for the same 1000 steps. At 2000 steps the gain is gone, and 1000 more steps of plain training gain as much on their own. The satellite branch was under-trained in those runs, so training it properly is the next test.
- **One WN3 feature didn't help at 1°:** epistemic dropout, because the fine-tuned model already has enough spread (§3). Hourly training leaves the 6-hourly forecast unchanged to 2% worse, a cost WN3 also reports.

The data, code and evaluation are now in place for each feature. They carry over to the 0.25° model, which is the next stage.

## 1. Setup

| | |
|---|---|
| Model | WN2 Mini (`WeatherNextCyclones_Mini`), 1° grid, 6 h step, 8-member ensemble (FGN noise) |
| Training data | GDAS analyses, 13 pressure levels, 2021-04 to 2023-12 |
| Test data | 2024-04 to 2025-03, never used in training or model selection; 52 start times, 10-day forecasts |
| Main score | Fair ensemble CRPS, latitude-weighted; lower is better. Results are given as % change, so negative = better. |
| Independent truth | SYNOP/METAR stations: a random 5% of station IDs held out from all training |
| Compute | NOAA RDHPCS Ursa, H100 GPUs; a 1000-step fine-tune takes about 2 h on 2 GPUs |

The WN2 model code is never edited. Every change is a patch applied when the model loads, which raises an error if the upstream code it relies on changes. So the work can follow new WN2 releases.

## 2. WN3 features and their 1° status

| WN3 feature | What we built | 1° result | Status |
|---|---|---|---|
| Fine-tuning on the target analysis | GDAS data pipeline, training with 1–8-step rollouts, evaluation scored against the analysis and held-out stations | CRPS vs released WN2: z500 −40% at 24 h and −11% at 10 days; 2 m T −45% / −21%; stations −5% to −7% at 3–10 days | Done |
| Station head (2 m T, Td) | Small network on the frozen model's internal fields, interpolated to any location; trained on GSI surface obs in sensible temperature | Held-out stations, 6–48 h: CRPS −35% to −41%, MAE −26% to −30%; bias from up to −0.9 K to within ±0.2 K; dewpoint CRPS −23% to −42% | **Done; strongest result** |
| Native hourly outputs | 55 new output channels (11 fields × 5 hours) trained on GDAS f001–f005; decoder trained first, then end to end | Hourly CRPS vs interpolation: z500 −33% at 1 h, msl −32%, 2 m T −15%; 6 h frames unchanged to +2% | **Done** |
| Hourly initialisation | Start the forecast at T+1…T+5 h from GDAS f001–f005 full states | No retraining: 6 h precipitation −6% to −15% at 6–24 h (better on 26/26 starts); z500 −29% to −50% | **Done** |
| Geostationary satellite input | NESDIS GMGSI mosaic (4 channels, hourly) at 1°, with its own encoder and decoder; images also predicted ahead | At 1000 fine-tune steps, vs a model with the same steps: q100 −25% to −28%, q300–q700 −3% to −5%, 2 m T −2% to −4.5% at 3–10 days. At 2000 steps: no gain (q100 +0.5%) | Built; passes at 1000 steps; gain not yet shown on a longer-trained model |
| Epistemic dropout | Dropout in the transformer, one mask per training step, independent masks per member at inference | CRPS +5% to +34% at 24 h: the 1° model already has too much spread there | Built; not helpful at 1° |
| Pooled precipitation CRPS, output clipping | Extra loss on each member's global-mean precipitation; non-negative precipitation and humidity | Tested only together with dropout | Built |
| Precipitation-observation head (IMERG) | — | — | Deferred |
| 0.1° fine grid, larger model, cloud and radiation outputs | — | — | Planned at 0.25° |

## 3. Results in more detail

### Station head

A small network (4.4 M parameters) reads the frozen forecast model's internal fields, interpolates them to each station, and adds station information: elevation, difference from model terrain, land/sea, time of day. It predicts a correction and a spread on top of the forecast 2 m temperature, interpolated in time and adjusted for height. That baseline is the best estimate the gridded forecast alone gives.

Test year, held-out stations, about 173,000 reports per lead:

| Lead | CRPS change | MAE change | Bias, head / baseline (K) | Spread/skill, head / baseline |
|---|---|---|---|---|
| 6 h | −41% | −27% | +0.05 / −0.36 | 0.67 / 0.21 |
| 24 h | −37% | −28% | −0.05 / −0.60 | 0.85 / 0.46 |
| 48 h | −38% | −30% | −0.18 / −0.94 | 0.95 / 0.55 |

Spread/skill is 1 for a well-calibrated ensemble. Stations above 1 km gain the most (CRPS −34% to −47%), since a grid-cell value is least representative in terrain. Marine reports gain less (−22% to −30%) because they are already assimilated into the analysis.

### Hourly outputs

For each 6-hourly step, the model also predicts 2 m T, 10 m wind, sea-level pressure, hourly precipitation, and a few upper-air fields for each hour in between. The first try, training everything together, made the 6-hourly forecast worse (z500 +32% at 48 h). Training the new output layer first, then everything with smaller hourly weights, fixed this:

| Hour after start | z500 | Sea-level pressure | 2 m T |
|---|---|---|---|
| 1 h | −33% | −32% | −15% |
| 5 h | −22% | −21% | −9% |

The values are CRPS change vs linear interpolation between 6-hourly frames. The gain lasts into day 2 (−5% to −11% for z500 and msl). Against a model trained for the same number of steps, the 6-hourly frames are unchanged to 2% worse. WN3 also reports a small 6 h loss.

### Hourly initialisation

WN3 starts a forecast every hour from short-range forecasts between analyses. We did the same with GDAS forecasts at 1–5 h. Using the model trained on analyses only, we compared a start at T+k with the forecast from T at the same valid times (read from its hourly outputs):

| Start | 6 h precipitation, 6–24 h | 6 h precipitation, 30–120 h |
|---|---|---|
| T+1 h | −6.3% | −1.3% |
| T+3 h | −14.6% | −3.1% |
| T+5 h | −8.6% | −1.6% |

The 95% confidence intervals are within ±0.6 points, and every test start is better. The forecast from a GDAS 1–5 h state is nearly as skilful as one from the analysis. So hourly starts are possible with no new training.

### Satellite input

The 1° satellite data are 4 GMGSI channels, hourly, 2021–2025 (39 GB). Building them meant finding and masking several kinds of bad data: fill values, missing sectors written as valid counts, and sectors holding the wrong image. A separate satellite encoder feeds the six hourly images before each start into the model. A gate makes it start out exactly equal to the model without satellite.

After 1000 fine-tune steps, the satellite model is better almost everywhere from 24 h than a model without satellite trained for the same 1000 steps, by the most for upper-air humidity:

| Variable | 72 h | 240 h |
|---|---|---|
| q100 | −25% | −28% |
| q700 | −5% | −4% |
| 2 m T | −4.5% | −2% |
| 6 h precipitation | −2.4% | −1.4% |

These match WN3, whose largest gains are also in humidity. 65–90% of the q100 gain is a smaller moist drift at 100 hPa rather than new detail. A version trained with the forecast model frozen, so only the satellite modules learn, keeps most of the gain. So at 1000 steps the gain comes from the satellite information, not from extra training. But 1000 more steps of plain training remove the drift just as well (q100 −30% to −39%). Adding satellite to a model trained for 2000 steps, compared with the same model without it, gives nothing so far: q100 +0.5%, other fields within ±0.6%. Its satellite branch barely trained at the low learning rate used there, though. The planned fix (next steps) trains the satellite modules first, as the frozen-model version did.

### What didn't help, and what we learned

- **Epistemic dropout:** the fine-tuned 1° model already has too much spread at 1–3 days, so more spread made CRPS worse. Station temperature is the exception: it is under-spread, which the station head fixes directly.
- **Training length:** plain training gains up to about 2000 steps after the 8-step rollout stage, then levels off on the grid; station 2 m T keeps improving to 4000. Every feature was compared against a model trained for the same number of steps, so its gain isn't just extra training.
- **Staged training:** new outputs and new inputs work best when the new layers are trained first with the model frozen, then end to end. Training them all together from the start hurt both hourly outputs and satellite.

## 4. What is now in place

- **Data**, all on the same 1° grid as training, 2021-04 to 2025-03:
  - GDAS hourly forecasts f001–f005 (212 GB);
  - GDAS f001–f005 full model states for hourly starts (880 GB);
  - GMGSI satellite imagery with quality control (39 GB);
  - GSI surface observations converted to sensible temperature, with GSI's own per-report flag.
- **Code:**
  - training with selective freezing;
  - the station head;
  - sub-time output channels;
  - the satellite encoder and decoder;
  - hourly-start loading.

  Each is a switch in a config file. 73 unit tests run on CPU.
- **Evaluation:**
  - fair CRPS, RMSE, bias and spread/skill against the analysis and held-out stations;
  - paired bootstrap confidence intervals;
  - hourly and hourly-start scoring;
  - scorecards.

  The same harness scores the 0.25° model.

## 5. Next steps

1. **Move to 0.25°.** The 0.25° fine-tune is trained; score it with the same harness, then carry the station head, hourly outputs and hourly initialisation over. An earlier 0.25° station head was already 6–8% better than the 1° one.
2. **Satellite, trained in two stages,** on the longer-trained model: satellite modules first, then end to end. Compare against a model trained for the same steps.
3. **Station head on the better-trained model,** and more spread at 6–18 h.
4. **Remaining WN3 pieces:** the IMERG precipitation head, cloud and radiation outputs, and the 0.1° grid for surface fields.
