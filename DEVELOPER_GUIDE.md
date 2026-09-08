# Mapping Heat Developer Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Key Components](#key-components)
4. [Data Pipeline](#data-pipeline)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

Mapping Heat is composed of three primary layers:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│   Frontend      │─────▶│   Backend API   │─────▶│   ML Model      │
│   (HTML/JS)     │      │   (Flask)       │      │   (LightGBM)    │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
  User Interface          SQLite Database         Trained Artifacts
  (Zone selection,        (Historical pitches,    (Model weights,
   probability viz,        player stats)           encoders, league
   pitch filtering)                                averages)
```

### Technology Stack

- Frontend: Vanilla JavaScript, D3.js for visual rendering, jQuery for HTTP requests.
- Backend: Python Flask REST server with an SQLite database for pitch history.
- Pipeline: Data processing with pandas, pybaseball, LightGBM, and scikit-learn.
- Deployment: Docker containerization managed via docker-compose.

---

## Project Structure

```
MapppingHeat/
├── backend/
│   ├── app.py                 # Flask API server and endpoint definitions
│   ├── model.py               # Model wrapper and inference logic
│   ├── pipeline.py            # Data fetching and model training pipeline
│   ├── db.py                  # Database initialization and SQL queries
│   ├── stats.py               # Statistical calculations and matchup blueprints
│   ├── schema.sql             # Database table structure
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container configuration
│   └── artifacts/             # Generated model artifacts and player profiles
│       ├── lgbm_model.pkl
│       ├── batters.json
│       ├── pitchers.json
│       ├── batter_stances.json
│       ├── pitcher_pitch_profiles.json
│       ├── pitcher_repertoire.json
│       └── league_averages.json
│
├── frontend/
│   ├── index.html             # D3.js single-page application interface
│   ├── nginx.conf             # Web server reverse proxy configuration
│   └── Dockerfile             # Frontend container configuration
│
├── gifs/                      # User interface demonstration animations
├── docker-compose.yml         # Container orchestrator configuration
└── Makefile                   # Utility commands for development tasks
```

---

## Key Components

### 1. Pipeline (`backend/pipeline.py`)

Downloads Statcast pitch data, processes features, trains the LightGBM model, and exports model artifacts.

Main Class: `MappingHeatPipeline`

Key Methods:
- `fetch_season_stats(years)`: Downloads player batting and pitching season statistics from FanGraphs.
- `fetch_statcast_bulk()`: Downloads pitch kinematics and outcomes from MLB Statcast.
- `fetch_id_mapping()`: Maps player identifiers between MLBAM IDs and FanGraphs IDs.
- `merge_context()`: Combines pitch data with player season statistics.
- `engineer_features()`: Generates count context, physics interaction, sequence, and player profile features.
- `train_model()`: Trains the LightGBM classifier with unweighted class distribution to preserve calibration.
- `calculate_pitch_profiles()`: Calculates average velocity, movement, and release point characteristics per pitcher.
- `save_artifacts()`: Exports model weights, encoders, rosters, profiles, and league averages.

### 2. Inference Engine (`backend/model.py`)

Loads artifacts from disk and computes live hit probabilities.

Main Class: `MappingHeatModel`

Key Methods:
- `load_artifacts()`: Reads trained LightGBM models, encoders, and JSON configuration files.
- `get_batter_stats(name)`: Retrieves batter profile metrics, falling back to league average defaults when unlisted.
- `get_pitcher_stats(name)`: Retrieves pitcher profile metrics, falling back to league average defaults when unlisted.
- `predict(pitch_data, batter, pitcher)`: Constructs feature vectors and calculates hit probability.

### 3. API Server (`backend/app.py`)

Flask application exposing endpoints for predictions, rosters, and player profiles.

Key Endpoints:
- `/rosters`: Returns rosters, stances, and player ID mappings.
- `/predict`: Calculates hit probability for a specific pitch scenario.
- `/pitch-profile`: Returns average pitch characteristics for a pitcher.
- `/player-stats`: Returns player statistics and pitch mix repertoire percentages.

### 4. Database Layer (`backend/db.py` and `backend/stats.py`)

SQLite connection management and querying for historical pitch history.

Key Functions:
- `get_db_connection()`: Creates an SQLite database connection.
- `init_db()`: Initializes table schema and loads Statcast CSV data.
- `query_pitcher_pitches(name)`: Retrieves historical pitch records for a pitcher.
- `query_pitcher_pitches_by_id(pitcher_id)`: Retrieves historical pitch records using an MLBAM ID.

---

## Data Pipeline

```
1. Fetch Season Stats (FanGraphs)
   ├── Batting stats (wOBA, OBP, SLG, BB%, K%, ISO, HardHit%, Barrel%)
   └── Pitching stats (FIP, ERA, WHIP, K/9, BB/9, xFIP, xERA, SwStr%)

2. Fetch Player ID Mappings (Chadwick)
   └── Map MLBAM IDs to FanGraphs IDs

3. Fetch Statcast Pitch Data (MLB)
   └── Download pitch velocity, movement, location, and results

4. Merge Context & Feature Engineering
   ├── Join pitch records with player metrics
   ├── Calculate count context (hitter's count, pitcher's count, leverage, two strikes)
   ├── Calculate physics interactions (speed x location, speed x height, velocity differential)
   └── Calculate sequence metrics (pitch distance delta, speed delta from prior pitch)

5. Train LightGBM Model & Export Artifacts
   └── Save model weights (lgbm_model.pkl), profiles, repertoires, and rosters
```

---

## API Reference

### GET /rosters

Returns available batters and pitchers along with stance classifications and ID mappings.

Response:
```json
{
  "batters": [{"name": "Aaron Judge", "display_name": "Judge, Aaron"}],
  "pitchers": [{"name": "Gerrit Cole", "display_name": "Cole, Gerrit"}],
  "batter_stances": { "Aaron Judge": "R" },
  "batter_ids": { "Aaron Judge": 592450 },
  "pitcher_ids": { "Gerrit Cole": 543037 }
}
```

---

### POST /predict

Calculates hit probability for a pitch scenario incorporating physics, sequence, and count context.

Request Body:
```json
{
  "batter": "Aaron Judge",
  "pitcher": "Gerrit Cole",
  "pitch_data": {
    "pitch_type": "FF",
    "zone": 5,
    "stand": "R",
    "p_throws": "R",
    "release_speed": 97.5,
    "balls": 2,
    "strikes": 2,
    "outs_when_up": 1,
    "pfx_x": 0.5,
    "pfx_z": 1.2,
    "prev_pitch_type": "SL",
    "prev_release_speed": 88.0,
    "prev_plate_x": -0.4,
    "prev_plate_z": 1.8,
    "pitch_number": 3
  }
}
```

Response:
```json
{
  "probability": 0.285,
  "batter": "Aaron Judge",
  "pitcher": "Gerrit Cole",
  "pitch_type": "FF"
}
```

---

### GET /player-stats

Returns statistics and pitch repertoire percentages for a batter or pitcher.

Query Parameters:
- `player`: Player name (e.g. "Gerrit Cole")
- `type`: Player type (`pitcher` or `batter`)

Response:
```json
{
  "player": "Gerrit Cole",
  "type": "pitcher",
  "stats": { "FIP": 3.42, "ERA": 3.41, "K/9": 9.6, "SwStr%": 0.124 },
  "repertoire": { "FF": 54.2, "SL": 24.1, "KC": 11.5, "CH": 10.2 }
}
```

---

### GET /stats/matchup-at-bats

Returns historical at-bats for a pitcher-batter pairing.

Query Parameters:
- `pitcher_id`: MLBAM Pitcher ID (e.g. `543037`)
- `batter_id`: MLBAM Batter ID (e.g. `646240`)

Response:
```json
[
  {
    "game_date": "2024-06-08",
    "inning": 3,
    "outs_when_up": 1,
    "pitch_count": 5,
    "final_event": "home_run"
  }
]
```

---

### GET /stats/at-bat-pitches

Returns chronological pitch records for a specific matchup at-bat.

Query Parameters:
- `pitcher_id`: MLBAM Pitcher ID
- `batter_id`: MLBAM Batter ID
- `game_date`: Date string in `YYYY-MM-DD` format
- `inning`: Inning number
- `outs_when_up`: Outs count

---

### GET /stats/pitcher

Returns historical pitch data for a pitcher by name or ID.

Query Parameters:
- `name`: Pitcher name (e.g. "Gerrit Cole")
- `id`: Optional MLBAM Pitcher ID

---

## Troubleshooting

### Connection failures to backend service

Check if the Flask server is running on port 5001 or inspect Docker container status:
```bash
docker-compose ps
docker-compose logs backend
```

### Unmatched pitcher records in database queries

The database stores player names in `Last, First` format while the API interface converts query inputs to `First Last`. Ensure name parameters match roster key formats specified in `db.py`.

### Model prediction mismatch across feature versions

When adding features to `engineer_features()` in `pipeline.py`, update `predict()` in `model.py` so the feature vector order matches the trained LightGBM feature names. Re-run `python backend/pipeline.py` to regenerate `lgbm_model.pkl`.
