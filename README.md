# Mapping Heat ⚾

## Description
Mapping Heat is an interactive web application for exploring major league pitching performance. It provides a dynamic heatmap that visualizes the predicted probability of a hit for any given pitch. This allows fans, analysts, and baseball professionals to explore complex pitcher-batter scenarios.

This tool uses a machine learning model (LightGBM) trained on recent Statcast data (2023-2025 Seasons) to generate its predictions. The frontend is built with D3.js and provides a comprehensive set of controls. These controls range from pitch type and speed to game situation and detailed pitch physics. They allow users to see how each variable impacts the probability of a hit in every part of the strike zone.

**Key Features:**
- 🎯 Real-time hit probability predictions using ML
- 📊 Interactive strike zone heatmap (14 zones)
- 📈 Historical pitch data visualization with outcome filtering
- 🔄 Automatic switch hitter detection and support
- 🎨 Dark/Light mode toggle
- 🎮 Full control over pitch characteristics and game situations

![pitch_outcome](./gifs/dark_mode_outcomes.gif)

![probability](./gifs/heat_map_prob.gif)

![pitch_selection](./gifs/pitcher_selection.gif)



## Model Details

This section outlines the process of developing the machine learning model that powers Mapping Heat.

#### Model Selection

The initial prototype of this project used a Logistic Regression model for its simplicity and interpretability. However, to achieve a higher level of predictive accuracy for complex baseball events, I transitioned to more powerful tree-based ensemble models.

After experimenting with both Random Forest and gradient boosting methods, LightGBM (Light Gradient Boosting Machine) was chosen as the final model. I selected LGBM for its high performance, training speed, and efficiency, which makes it ideal for handling the large and feature-rich Statcast dataset.

#### Training Process & Feature Engineering

The model is trained on a comprehensive dataset to ensure its predictions are relevant and accurate.

###### Data Source: 
The training data consists of all pitches from the 2023 MLB season through the 2024 regular season, fetched using the pybaseball library.

###### Target Variable: 
The model predicts the probability of a hit. A hit is strictly defined as an event where the outcome is a single, double, triple, or home run. This provides a more accurate and less ambiguous target than other descriptive fields.

###### Features: 
The model uses a rich set of over 25 features, including:
- Pitch Kinematics: release_speed, release_spin_rate, pfx_x, pfx_z, release_extension, effective_speed, spin_axis, and release point.

- Game Context: The count (balls and strikes), outs, inning, and which bases are occupied.

- Situational Details: Pitcher and batter stance (L/R), pitch type, and the pitcher's throwing hand.

- Preprocessing: Before training, the data is processed through a pipeline that applies one-hot encoding to categorical features and uses a StandardScaler to normalize all numerical features.

#### Experiments & Performance Improvements
The final model is the result of several iterative experiments aimed at increasing predictive power:

###### Improved Data: 
The first major improvement came from replacing the original 2020 dataset with fresh data from 2023 through 2024. This ensures the model is trained on recent player performance and league trends.

###### Refined Target: 
Switching the prediction target from the ambiguous description field to the concrete events field provided a much cleaner signal for the model to learn what constitutes a hit.

###### Advanced Models: 
Moving from a linear model (Logistic Regression) to an advanced gradient boosting model (LGBM) captured the complex, non-linear interactions between pitch features, significantly boosting accuracy.

###### Feature Expansion: 
The most significant performance gain came from expanding the feature set. By including detailed pitch physics (like spin axis and release point) and more game context, the model was able to create a much more nuanced and accurate prediction.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Btlyons1/MappingHeat
cd MapppingHeat

# Start the application
docker-compose up --build

# Open in browser
open http://localhost:8080
```

### Option 2: Manual Setup

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt

# 2. Run the data pipeline (fetches data and trains model)
python pipeline.py

# 3. Start the backend API
python app.py

# 4. Open frontend
# Open frontend/index.html in your browser
```

## Project Structure

```
MapppingHeat/
├── backend/              # Flask API and ML pipeline
│   ├── app.py           # REST API server
│   ├── pipeline.py      # Data fetching & model training
│   ├── model.py         # ML model wrapper
│   ├── db.py            # Database queries
│   └── artifacts/       # Trained model and data
├── frontend/            # Web interface
│   └── index.html       # Single-page app
├── model/               # Development notebooks
│   ├── model_evaluation.ipynb    # Model metrics & analysis
│   └── new_model_batters.ipynb   # Model development
└── docker-compose.yml   # Multi-container setup
```

## Documentation

- **[Developer Guide](DEVELOPER_GUIDE.md)** - Complete guide for adding features and understanding the codebase
- **[Model Evaluation Notebook](model/model_evaluation.ipynb)** - ROC, precision, recall, and other ML metrics

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rosters` | GET | Get available batters, pitchers, and stances |
| `/predict` | POST | Predict hit probability for a pitch |
| `/pitch-profile` | GET | Get pitcher's average pitch characteristics |
| `/stats/pitcher` | GET | Get historical pitches for a pitcher |

**Examples:**

```bash
# Get pitcher stats
curl "http://127.0.0.1:5001/stats/pitcher?name=Gerrit%20Cole" | jq .

# Get pitch profile
curl "http://127.0.0.1:5001/pitch-profile?pitcher=Gerrit%20Cole&pitch_type=FF" | jq .

# Get rosters
curl "http://127.0.0.1:5001/rosters" | jq .
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#api-reference) for detailed API documentation.
