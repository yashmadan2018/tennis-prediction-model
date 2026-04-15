# Tennis Prediction Model

[![Quarterly Retrain](https://github.com/yashmadan2018/tennis-prediction-model/actions/workflows/retrain.yml/badge.svg)](https://github.com/yashmadan2018/tennis-prediction-model/actions/workflows/retrain.yml)

Pre-match win probability model for ATP Tour, WTA Tour, and ATP Challenger events.

See [CLAUDE.md](CLAUDE.md) for the full project specification.

---

## Model

Ensemble of XGBoost + Logistic Regression + MLP (scikit-learn).

| Metric | Value |
|--------|-------|
| Val Brier (ensemble, 2022–23) | 0.2115 |
| Val Brier (XGBoost only)      | 0.2093 |
| Test Brier (2024, held out)   | 0.2152 |
| Test accuracy (≥65% conf)     | 74.7%  |

The model retrains automatically every quarter (Jan/Apr/Jul/Oct) via GitHub Actions.
New models only replace the saved version if the validation Brier improves.

---

## Quickstart

```bash
pip install -r requirements.txt

# Download Sackmann data
git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp
git clone https://github.com/JeffSackmann/tennis_wta.git data/raw/tennis_wta

# Process data and train
python utils/data_loader.py
python models/train.py --rebuild
python models/ensemble.py

# Run predictions (requires ODDS_API_KEY in .env)
python predict.py
```

---

## Streamlit App

```bash
streamlit run app.py
```

Four pages: **Daily Slate** · **Match Deep Dive** · **Model Performance** · **Settings**

Deploy to Streamlit Cloud — add `ODDS_API_KEY` and Twilio credentials in App Settings → Secrets.

---

## SMS Alerts

Edge alerts are sent automatically after each prediction run:

```bash
# Dry-run (prints to terminal)
python -m utils.alert_runner --dry-run

# Live send
python -m utils.alert_runner
```

Configure credentials in `.env` (local) or Streamlit Cloud secrets — see `.streamlit/secrets.toml.example`.

---

## Quarterly Retrain

The workflow in `.github/workflows/retrain.yml` runs automatically on the 1st of
January, April, July, and October at 06:00 UTC.

It can also be triggered manually:
**Actions → Quarterly Retrain → Run workflow**

The retrain log is written to `output/retrain_log.json` after each run.

---

## Project Structure

```
tennis-prediction-model/
├── app.py                      Streamlit app (4 pages)
├── predict.py                  End-to-end prediction runner
├── requirements.txt
├── .github/workflows/
│   └── retrain.yml             Quarterly retraining workflow
├── .streamlit/
│   ├── config.toml             Dark theme
│   └── secrets.toml.example   Credentials template
├── data/
│   ├── raw/                    Sackmann CSVs (not committed)
│   └── processed/              Cleaned match data (not committed)
├── features/
│   ├── elo.py                  Surface-specific Elo
│   ├── serve_return.py         Serve/return stats
│   ├── form.py                 Recent form
│   ├── h2h.py                  Head-to-head
│   ├── matchup.py              Playing style matchup
│   ├── context.py              Fatigue, venue history
│   ├── injury.py               Injury proxies
│   ├── market.py               Odds / CLV
│   └── pipeline.py             Assembles all features
├── models/
│   ├── train.py                XGBoost training (--rolling flag)
│   ├── ensemble.py             LR + MLP ensemble
│   ├── confidence.py           Distance-ratio CI (SHARP/MODERATE/WIDE)
│   ├── evaluate.py             Test-set evaluation
│   ├── retrain.py              Quarterly retrain orchestrator
│   └── saved/                  Trained model pkl files (committed)
├── output/
│   ├── predictions.csv         All predictions with full metadata
│   ├── clv_tracker.csv         CLV log
│   └── retrain_log.json        Retrain audit trail
└── utils/
    ├── data_loader.py          Load + merge Sackmann CSVs
    ├── odds_fetcher.py         Odds API integration
    ├── slate_generator.py      Tomorrow's match slate
    ├── alerts.py               Twilio SMS sender
    ├── alert_runner.py         Edge alert runner (dedup via sent_alerts.csv)
    ├── result_logger.py        Post-match result backfill
    └── court_speed.py          Court speed index
```
