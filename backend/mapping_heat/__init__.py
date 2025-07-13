# === __init__.py ===

import os
from typing import Optional, Dict, Any
from flask import Flask
from flask_cors import CORS
from . import stats
from . import db
from . import model



AppType = Flask

def create_app(test_config: Optional[Dict[str, Any]] = None) -> AppType:
    """Create and configure an instance of the Flask application."""
    print("Creating Flask app instance...")
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app) # Enable CORS for all routes
    app.config.from_mapping(
        SECRET_KEY='dev', 
        DATABASE=os.path.join(app.instance_path, 'mapping_heat.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        # Use app.instance_path which points to the 'instance' folder
        config_path = os.path.join(app.instance_path, 'config.py')
        app.config.from_pyfile(config_path, silent=True)
        if os.path.exists(config_path):
             print(f"Loaded instance config from {config_path}")
        else:
             print(f"Instance config {config_path} not found, using defaults.")
    else:
        # load the test config if passed in
        print("Loading test configuration.")
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
        print(f"Instance path {app.instance_path} ensured.")
    except OSError:
        # Already exists or other error, ignore for now
        pass

    # Initialize database
    print("Initializing database...")
    db.init_app(app)
    print("Database initialized.")

    # Initialize model on startup within application context
    print("Initializing ML model...")
    with app.app_context():
        # This ensures necessary context like 'current_app' is available if needed
        model.init_model() # Load model, scaler, feature names
        print('ML Model initialization attempted.')

    # Register blueprints
    print("Registering blueprints...")

    app.register_blueprint(stats.bp)
    print("Stats blueprint registered.")

    print("Flask app creation complete.")
    return app