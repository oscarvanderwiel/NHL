# NHL Forecasting Pipeline

A Python-based NHL game outcome forecasting system using team ELO ratings and statistical models (Multinomial Logit and Ordered Probit).

## Overview

This project analyzes NHL game data to predict match outcomes using:
- **Step 1**: Data cleaning and preparation
- **Step 2**: Team ELO rating calculation
- **Step 3**: Match outcome forecasting using:
  - Multinomial Logit model
  - Ordered Probit model

## Requirements

- Python 3.8 or higher
- Required packages (see `requirements.txt`):
  - pandas
  - numpy
  - statsmodels

## Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd NHL
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files are present**:
   - `gamedata.csv` - Game-level data
   - `playergamedata.csv` - Player-level data

   If `playergamedata.csv` is missing, extract it from the zip file:
   ```bash
   unzip playergamedata.zip
   ```

## How to Run

### Run the Full Pipeline

To run all three steps (data cleaning, ELO ratings, and forecasting models):

```bash
python main.py
```

This will:
1. Load and clean player and game data
2. Calculate team ELO ratings
3. Run both forecasting models (Multinomial Logit and Ordered Probit)
4. Cache intermediate results in a `data/` directory for faster subsequent runs

### Run Individual Steps

You can also run each step independently:

**Step 1 - Data Cleaning**:
```bash
python step1_clean_data.py
```

**Step 2 - ELO Ratings**:
```bash
python step2_ELO_ratings.py
```

**Step 3 - Multinomial Logit**:
```bash
python step3_multinomial_logit.py
```

**Step 3.1 - Ordered Probit**:
```bash
python step3_ordered_probit.py
```

### Using the Cache

The pipeline caches processed data to speed up subsequent runs. By default, `main.py` uses cached data if available.

To force a fresh run (ignoring cache):
- Edit `main.py` line 101 and change:
  ```python
  main(use_cache=False)
  ```

Cached files are stored in the `data/` directory:
- `clean_player.csv`
- `clean_games.csv`
- `delta_df.csv`
- `player_FE.csv`

## Output

The pipeline produces:
- **ELO ratings** for all teams
- **Delta values** (home team advantage metric)
- **Predicted probabilities** for match outcomes:
  - P(home win)
  - P(tie)
  - P(away win)
- **Model fit statistics** from both forecasting models

## Project Structure

```
NHL/
├── main.py                      # Main pipeline orchestrator
├── step1_clean_data.py          # Data cleaning
├── step2_ELO_ratings.py         # ELO rating calculation
├── step3_multinomial_logit.py   # Multinomial logit model
├── step3_ordered_probit.py      # Ordered probit model
├── gamedata.csv                 # Game data
├── playergamedata.csv           # Player data (extracted from zip)
├── playergamedata.zip           # Compressed player data
├── requirements.txt             # Python dependencies
└── data/                        # Cached outputs (created on first run)
```

## Troubleshooting

**ImportError**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**FileNotFoundError**: Ensure `gamedata.csv` and `playergamedata.csv` are in the project directory.

**Memory issues**: The player data file is ~64MB. Ensure you have sufficient RAM available.

## Notes

- First run may take several minutes as ELO ratings are computed for all games
- Subsequent runs are much faster when using cached data
- The models predict regulation-time outcomes (excluding overtime/shootouts)
