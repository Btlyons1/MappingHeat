# Mapping Heat

## Description
Mapping Heat is an interactive web application for exploring major league pitching performance. It provides a dynamic heatmap that visualizes the predicted probability of a hit for any given pitch. This allows fans, analysts, and baseball professionals to explore complex pitcher-batter scenarios.

This tool uses a machine learning model (LGBM) trained on recent Statcast data (2023-2024 Seasons) to generate its predictions. The frontend is built with D3.js and provides a comprehensive set of controls. These controls range from pitch type and speed to game situation and detailed pitch physics. They allow users to see how each variable impacts the probability of a hit in every part of the strike zone.

The project includes:
- A Jupyter Notebook for data fetching, feature engineering, and model training.
- A Flask backend that serves the ML model's predictions via a REST API.
- An interactive D3.js frontend for data visualization and user control.

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

## Installation
- Download entire package from GitHub
- `cd` into backend directory
- Go to the 'instance' folder and unzip mapping_heat.sqlite.zip
- Go to the 'fixtures' folder and unzip pitching_data.csv.zip
- Ensure you have unzipped the proper files before moving on
- Ensure you all the proper packages, 'pip install -r requirements.txt'
- Run `flask init-db` to initialize the SQLite data base
- After these steps, the package is downloaded and ready to return

## Execution
- From the backend folder, 'export FLASK_APP=mapping_heat'
- From the backend folder, 'export FLASK_ENV=development'
- Run `flask run` which will launch the backend process
- In a separate terminal, launch a local HTTP server via 'python3 -m http.server 8000'
- Open 'http://0.0.0.0:8000/pitch_v1.html'in a browser of choice and the visual should be live



### Hit endpoints
- pitcher endpoint
  -  ` curl  http://127.0.0.1:5000/stats/pitcher?name=Brad%20Hand | jq .`
- pitch type endpoint
  - `curl  "http://127.0.0.1:5000/stats/pitcher/pitch?name=Brad%20Hand&pitch=FF" | jq .`
- pitch feature endpoint
  - ` curl "http://127.0.0.1:5000/stats/pitcher?name=Brad%20Hand&feature=release_speed&value=78.7" | jq .`
