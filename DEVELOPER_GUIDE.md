# Mapping Heat - Developer Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Key Components](#key-components)
4. [Data Pipeline](#data-pipeline)
5. [Adding New Features](#adding-new-features)
6. [Common Tasks](#common-tasks)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Mapping Heat is a baseball pitch prediction application with three main components:

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
  - Zone selection        - Historical pitches    - Model weights
  - Probability viz       - Player stats          - Encoders
  - Pitch filtering                               - League averages
  - Matchups                                             
```

### Technology Stack

- **Frontend**: Vanilla JavaScript, D3.js for visualizations, jQuery for AJAX
- **Backend**: Python Flask, SQLite for data storage
- **ML Pipeline**: pandas, pybaseball, LightGBM, scikit-learn
- **Deployment**: Docker, docker-compose

---

## Project Structure

```
MapppingHeat/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── model.py               # ML model wrapper & prediction logic
│   ├── pipeline.py            # Data fetching & model training pipeline
│   ├── db.py                  # Database initialization & queries
│   ├── stats.py               # Statistical calculations
│   ├── schema.sql             # Database schema
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Backend container config
│   └── artifacts/            # Generated model artifacts
│       ├── lgbm_model.pkl
│       ├── batters.json
│       ├── pitchers.json
│       ├── batter_stances.json
│       ├── pitcher_pitch_profiles.json
│       └── league_averages.json
│
├── frontend/
│   └── index.html            # Single-page application
│
├── model/
│   └── new_model_batters.ipynb  # Model development notebook
│
├── docker-compose.yml        # Multi-container orchestration
├── Makefile                  # Common development tasks
└── .env.example             # Environment variables template
```

---

## Key Components

### 1. Pipeline (`backend/pipeline.py`)

**Purpose**: Fetches baseball data, processes it, trains the model, and generates artifacts.

**Main Class**: `MappingHeatPipeline`

**Key Methods**:

- `fetch_season_stats(years)` - Downloads batting/pitching stats from Fangraphs
- `fetch_statcast_bulk()` - Downloads pitch-level data from MLB Statcast
- `fetch_id_mapping()` - Maps player IDs between systems (MLBAM ↔ Fangraphs)
- `merge_context()` - Joins pitch data with player season stats
- `engineer_features()` - Creates model features and target variable
- `train_model()` - Trains LightGBM classifier
- `calculate_pitch_profiles()` - Computes average pitch characteristics per pitcher
- `save_artifacts()` - Exports model, rosters, stances, profiles, averages

**When to modify**:
- Adding new data sources
- Changing feature engineering
- Updating model hyperparameters
- Adding new artifact exports

**Example - Adding a new feature**:
```python
# In engineer_features() method:
def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...

    # Add your new feature
    df['my_new_feature'] = df['existing_col1'] * df['existing_col2']

    return df
```

---

### 2. Model (`backend/model.py`)

**Purpose**: Loads trained artifacts and makes predictions.

**Main Class**: `MappingHeatModel`

**Key Methods**:

- `load_model()` - Loads LightGBM model and label encoders
- `load_rosters()` - Loads batters, pitchers, stances from JSON
- `get_batter_stats(name)` - Retrieves batter stats or league average
- `get_pitcher_stats(name)` - Retrieves pitcher stats or league average
- `predict(pitch_data, batter, pitcher)` - Returns hit probability

**When to modify**:
- Adding new player stats to prediction
- Changing fallback logic for missing players
- Adding prediction confidence intervals
- Implementing new prediction endpoints

**Example - Adding a new stat to predictions**:
```python
# In predict() method:
def predict(self, pitch_data: Dict, batter_name: Optional[str] = None, ...):
    # ... existing code ...

    # Add new stat
    batter_stats = self.get_batter_stats(batter_name)
    features['batter_new_stat'] = batter_stats.get('NewStat', 0.0)

    # Make sure to train the model with this feature in pipeline.py!
```

---

### 3. API (`backend/app.py`)

**Purpose**: Flask REST API exposing model predictions and data.

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rosters` | GET | Returns batters, pitchers, batter_stances |
| `/predict` | POST | Predicts hit probability for pitch scenario |
| `/pitch-profile` | GET | Returns pitcher's average pitch characteristics |
| `/stats/pitcher` | GET | Returns historical pitches for a pitcher |
| `/player-stats` | GET | Returns player stats (batter or pitcher) |

**When to modify**:
- Adding new API endpoints
- Changing response formats
- Adding authentication
- Implementing caching

**Example - Adding a new endpoint**:
```python
@app.route('/team-stats', methods=['GET'])
def get_team_stats():
    """Get aggregated stats for a team."""
    team_name = request.args.get('team')

    # Your logic here
    team_data = calculate_team_stats(team_name)

    return jsonify({
        'team': team_name,
        'stats': team_data
    })
```

---

### 4. Database (`backend/db.py`)

**Purpose**: SQLite database management for historical pitch data.

**Key Functions**:

- `get_db_connection()` - Returns SQLite connection
- `init_db()` - Creates schema and loads CSV data
- `query_pitcher_pitches(name)` - Gets all pitches for a pitcher
- `query_pitcher_names()` - Lists all pitchers in database
- `query_batter_stances()` - Gets batter stance mappings

**When to modify**:
- Adding new database tables
- Creating new query functions
- Optimizing database indexes

**Example - Adding a new query**:
```python
def query_batter_pitches(batter_name: str):
    """Query all pitches faced by a specific batter."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM pitching_data WHERE batter_name = ? ORDER BY game_date DESC",
            (batter_name,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
```

---

### 5. Frontend (`frontend/index.html`)

**Purpose**: Interactive visualization and user interface.

**Key JavaScript Functions**:

- `loadPitchProfile()` - Loads pitcher's pitch characteristics
- `updateVisuals(zone)` - Requests predictions from API
- `drawChartElements(pitches, probs)` - Renders strike zone heatmap and pitch dots
- `filterPitchDots()` - Filters visible pitches by outcome and year
- `getOutcomeCategory(d)` - Categorizes pitch outcomes (Hit/Out/Strike/Ball)

**Key UI Elements**:

- Strike zone with 14 zones (1-9 in zone, 11-14 outside)
- Historical pitch dots colored by outcome
- Heatmap showing hit probabilities
- Control panel with pitch/situation inputs
- Outcome filters and year selector

**When to modify**:
- Adding new visualizations
- Changing UI layout
- Adding new controls
- Implementing new filtering options

**Example - Adding a new filter**:
```javascript
// In HTML:
<select id="inningFilter">
    <option value="all">All Innings</option>
    <option value="early">1-3</option>
    <option value="middle">4-6</option>
    <option value="late">7-9</option>
</select>

// In JavaScript:
function filterPitchDots() {
    let selectedInning = $('#inningFilter').val();

    svg.selectAll('.pitch-dot').style('display', function(d) {
        let inningMatch = true;
        if (selectedInning === 'early') {
            inningMatch = d.inning <= 3;
        } else if (selectedInning === 'middle') {
            inningMatch = d.inning >= 4 && d.inning <= 6;
        } else if (selectedInning === 'late') {
            inningMatch = d.inning >= 7;
        }

        return inningMatch ? '' : 'none';
    });
}
```

---

## Data Pipeline

### Pipeline Flow

```
1. Fetch Season Stats (Fangraphs)
   ├── Batting stats (wOBA, OBP, SLG, BB%, K%)
   └── Pitching stats (FIP, ERA, WHIP, K/9, BB/9)

2. Fetch Player ID Mappings (Chadwick)
   └── Map MLBAM IDs ↔ Fangraphs IDs

3. Fetch Statcast Data (MLB)
   ├── Pitch-level data (velocity, movement, location)
   └── Chunked by month to avoid timeouts

4. Merge Data
   ├── Join pitch data with season stats via IDs
   └── Calculate league averages

5. Feature Engineering
   ├── Create target variable (Hit vs Out)
   ├── Fill missing values
   └── Encode categorical variables

6. Train Model
   ├── LightGBM classifier
   └── 80/20 train/test split

7. Generate Artifacts
   ├── Model & encoders (lgbm_model.pkl)
   ├── Rosters (batters.json, pitchers.json)
   ├── Batter stances (batter_stances.json)
   ├── Pitch profiles (pitcher_pitch_profiles.json)
   └── League averages (league_averages.json)
```

### Running the Pipeline

```bash
# Basic run (uses default dates in pipeline.py)
python backend/pipeline.py

# Or modify dates in pipeline.py main block:
pipeline = MappingHeatPipeline(
    start_date="2024-04-01",  # Start of season
    end_date="2024-10-31",     # End of season
    output_dir="artifacts"
)
```

### Pipeline Caching

The pipeline caches intermediate results to speed up re-runs:

- **Season stats**: `cache/season_stats_{start_year}_{end_year}.pkl`
- **ID mappings**: `cache/id_mapping.pkl`
- **Statcast data**: `cache/statcast_{start_date}_to_{end_date}.pkl`

To force fresh data, delete the cache files:
```bash
rm -rf backend/cache/
```

---

## Adding New Features

### 1. Adding a New Player Stat

**Step 1**: Update pipeline to include the stat
```python
# In pipeline.py, modify batter_cols_to_keep or pitcher_cols_to_keep
batter_cols_to_keep = [
    'batter_fg_id', 'year', 'wOBA', 'OBP', 'SLG', 'BB%', 'K%',
    'ISO', 'BABIP', 'MyNewStat'  # Add here
]
```

**Step 2**: Update model to use the stat
```python
# In model.py, add to get_batter_stats() or get_pitcher_stats()
return {
    'wOBA': ...,
    'MyNewStat': self.league_averages.get('batter_MyNewStat', 0.0)
}

# In predict(), add feature
features['batter_MyNewStat'] = batter_stats.get('MyNewStat', 0.0)
```

**Step 3**: Re-train model
```bash
python backend/pipeline.py
```

---

### 2. Adding a New Visualization

**Step 1**: Add HTML controls
```html
<div class="input-group">
    <label for="myNewControl">My Feature:</label>
    <select id="myNewControl">
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
    </select>
</div>
```

**Step 2**: Add JavaScript logic
```javascript
// Add event listener
$('#myNewControl').on('change', function() {
    updateMyVisualization();
});

function updateMyVisualization() {
    let value = $('#myNewControl').val();

    // Update D3 visualization
    svg.selectAll('.my-elements')
        .data(myData)
        .enter()
        .append('circle')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        // ... more D3 code
}
```

---

### 3. Adding a New API Endpoint

**Step 1**: Define endpoint in app.py
```python
@app.route('/my-endpoint', methods=['GET', 'POST'])
def my_endpoint():
    """
    Description of what this endpoint does.

    Args (via query params or JSON body):
        param1: Description
        param2: Description

    Returns:
        JSON response with data
    """
    try:
        # Get parameters
        param1 = request.args.get('param1') if request.method == 'GET' else request.json.get('param1')

        # Your logic
        result = process_data(param1)

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Error in my_endpoint: {e}")
        return jsonify({'error': str(e)}), 500
```

**Step 2**: Call from frontend
```javascript
$.get(`${baseApiUrl}/my-endpoint?param1=value`, function(data) {
    console.log('Got data:', data);
    updateUI(data);
});
```

---

## Common Tasks

### Updating the Model

1. Modify training logic in [pipeline.py](backend/pipeline.py)
2. Run pipeline: `python backend/pipeline.py`
3. Restart backend: `docker-compose restart backend`

### Adding a New Batter/Pitcher

Batters and pitchers are automatically added when you re-run the pipeline with updated data. To add manually:

1. Edit [batters.json](backend/artifacts/batters.json) or [pitchers.json](backend/artifacts/pitchers.json)
2. Add entry with required stats
3. Restart: `docker-compose restart backend`

### Changing Date Range

Edit [pipeline.py](backend/pipeline.py):
```python
pipeline = MappingHeatPipeline(
    start_date="2024-04-01",  # Change these
    end_date="2024-10-31",
    output_dir="artifacts"
)
```

### Debugging the Frontend

1. Open browser DevTools (F12)
2. Check Console for JavaScript errors
3. Check Network tab for API calls
4. Use `console.log()` for debugging (remove before commit)

### Debugging the Backend

1. Check Docker logs: `docker-compose logs backend`
2. Add logging: `logger.info("Debug message")`
3. For interactive debugging, run outside Docker:
   ```bash
   cd backend
   python app.py
   ```

---

## API Reference

### GET /rosters

Returns lists of available batters and pitchers.

**Response**:
```json
{
  "batters": ["Aaron Judge", "Shohei Ohtani", ...],
  "pitchers": ["Gerrit Cole", "Sandy Alcantara", ...],
  "batter_stances": {
    "Aaron Judge": "R",
    "Shohei Ohtani": "L",
    "Switch Hitter": "S"
  }
}
```

---

### POST /predict

Predicts hit probability for a pitch scenario.

**Request Body**:
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
    ...
  }
}
```

**Response**:
```json
{
  "probability": 0.285,
  "batter": "Aaron Judge",
  "pitcher": "Gerrit Cole",
  "zone": 5
}
```

---

### GET /pitch-profile

Returns average pitch characteristics for a pitcher's pitch type.

**Query Parameters**:
- `pitcher` - Pitcher name
- `pitch_type` - Pitch type code (FF, SL, CH, etc.)

**Response**:
```json
{
  "profile": {
    "release_speed": 96.8,
    "release_spin_rate": 2350,
    "pfx_x": 0.6,
    "pfx_z": 1.1,
    "arm_angle": 28.5,
    ...
  }
}
```

---

### GET /stats/pitcher

Returns historical pitch data for a pitcher.

**Query Parameters**:
- `name` - Pitcher name (e.g., "Gerrit Cole")

**Response**:
```json
[
  {
    "pitch_type": "FF",
    "game_date": "2024-09-15",
    "release_speed": 97.2,
    "plate_x": 0.3,
    "plate_z": 2.8,
    "events": "strikeout",
    "zone": 5,
    ...
  },
  ...
]
```

---

## Troubleshooting

### Issue: "Failed to connect to backend"

**Cause**: Backend not running or wrong port

**Solution**:
```bash
# Check if backend is running
docker-compose ps

# Restart backend
docker-compose restart backend

# Check logs
docker-compose logs backend
```

---

### Issue: "No data returned for pitcher"

**Cause**: Pitcher name doesn't match database format

**Solution**:
- Database stores: "Last, First" format
- UI expects: "First Last" format
- Check [db.py](backend/db.py) `query_pitcher_pitches()` for name conversion logic

---

### Issue: "Model predictions are all the same"

**Cause**: Model not properly loaded or features mismatched

**Solution**:
1. Check artifact files exist in `backend/artifacts/`
2. Verify model was trained with current feature set
3. Re-run pipeline: `python backend/pipeline.py`

---

### Issue: "Batter stances showing pitchers"

**Cause**: Old batter_stances.json with wrong data

**Solution**:
```bash
# Delete old file
rm backend/artifacts/batter_stances.json

# Re-run pipeline
python backend/pipeline.py
```

---

### Issue: "Docker build fails"

**Cause**: Missing dependencies or cache issues

**Solution**:
```bash
# Clear Docker cache and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## Best Practices

1. **Version Control**: Always commit working code before making major changes
2. **Testing**: Test changes locally before deploying
3. **Logging**: Use `logger.info()` for important events, `logger.error()` for errors
4. **Documentation**: Update this guide when adding features
5. **Data Validation**: Always validate user input in API endpoints
6. **Error Handling**: Use try/except blocks in Python, check for null/undefined in JS
7. **Performance**: Cache expensive operations, use database indexes
8. **Security**: Never commit API keys or credentials

---

## Development Workflow

```bash
# 1. Make changes to code
vim backend/app.py

# 2. Test locally (if not using Docker)
cd backend
python app.py

# 3. Rebuild and restart containers
docker-compose up --build

# 4. Test in browser
open http://localhost

# 5. Commit changes
git add .
git commit -m "Add feature X"
git push
```

---

## Additional Resources

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [D3.js Documentation](https://d3js.org/)
- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [MLB Statcast Search](https://baseballsavant.mlb.com/statcast_search)
- [Fangraphs](https://www.fangraphs.com/)

---
