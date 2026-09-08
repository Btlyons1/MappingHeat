"""
Mapping Heat Flask API - Hybrid Backend
Combines ML predictions with historical pitch visualization
"""
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Configure CORS to allow all origins (suitable for development)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize database
logger.info("Initializing Mapping Heat backend")

try:
    from db import init_db
    init_db()
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    logger.error("Continuing without database: historical pitch queries will not be available")

# Load ML model
artifacts_dir = '/app/artifacts'
if not os.path.exists(artifacts_dir):
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts'))
logger.info(f"Loading ML model from {artifacts_dir}")

try:
    import model as model_module
    MappingHeatModel = model_module.MappingHeatModel
    model = MappingHeatModel(artifacts_dir=artifacts_dir)
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# Register stats blueprint for legacy endpoints
try:
    from stats import bp as stats_bp
    app.register_blueprint(stats_bp)
    logger.info("Stats blueprint registered")
except Exception as e:
    logger.warning(f"Stats blueprint registration failed: {e}")
    logger.warning("Historical pitch data will not be available")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'mapping-heat-backend'}), 200


@app.route('/rosters', methods=['GET'])
def get_rosters():
    """
    Get available batters and pitchers for UI dropdowns.
    
    Returns:
        JSON with batters and pitchers lists
    """
    try:
        rosters = model.get_rosters()
        return jsonify(rosters), 200
    except Exception as e:
        logger.error(f"Error fetching rosters: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/pitch-profile', methods=['GET'])
def get_pitch_profile():
    """
    Get pitch characteristics for a specific pitcher and pitch type.
    
    Query params:
        - pitcher: Pitcher name
        - pitch_type: Pitch type code (FF, SL, etc.)
    
    Returns:
        JSON with pitch characteristics
    """
    try:
        pitcher_name = request.args.get('pitcher')
        pitch_type = request.args.get('pitch_type', 'FF')
        
        # Handle "Average" or empty selections
        if pitcher_name in ['Average', 'League Average', '', None]:
            pitcher_name = None
        
        profile = model.get_pitch_profile(pitcher_name, pitch_type)
        
        return jsonify({
            'pitcher': pitcher_name or 'League Average',
            'pitch_type': pitch_type,
            'profile': profile
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching pitch profile: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict probability of hit given pitch and matchup.
    
    Expected JSON body:
    {
        "batter": "Mike Trout" or null,
        "pitcher": "Gerrit Cole" or null,
        "pitch_data": {
            "release_speed": 95.5,
            "pfx_x": -5.2,
            ...
        }
    }
    
    Returns:
        JSON with probability prediction
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        batter_name = data.get('batter')
        pitcher_name = data.get('pitcher')
        pitch_data = data.get('pitch_data', {})
        
        # Handle "Average" or empty selections
        if batter_name in ['Average', 'League Average', '', None]:
            batter_name = None
        if pitcher_name in ['Average', 'League Average', '', None]:
            pitcher_name = None
        
        # Make prediction
        probability = model.predict(
            pitch_data=pitch_data,
            batter_name=batter_name,
            pitcher_name=pitcher_name
        )
        
        # Log prediction
        logger.info(
            f"Prediction - Batter: {batter_name or 'League Avg'}, "
            f"Pitcher: {pitcher_name or 'League Avg'}, "
            f"Prob: {probability:.3f}"
        )
        
        return jsonify({
            'probability': probability,
            'batter': batter_name or 'League Average',
            'pitcher': pitcher_name or 'League Average',
            'pitch_type': pitch_data.get('pitch_type', 'Unknown')
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/player-stats', methods=['GET'])
def get_player_stats():
    """
    Get stats for a specific player.
    
    Query params:
        - player: Player name
        - type: 'batter' or 'pitcher'
    
    Returns:
        JSON with player stats
    """
    try:
        player_name = request.args.get('player')
        player_type = request.args.get('type', 'batter')
        
        if not player_name:
            return jsonify({'error': 'Player name required'}), 400
        
        if player_type == 'batter':
            stats = model.get_batter_stats(player_name)
            return jsonify({
                'player': player_name,
                'type': player_type,
                'stats': stats
            }), 200
        elif player_type == 'pitcher':
            stats = model.get_pitcher_stats(player_name)
            
            # Retrieve repertoire (handling potential suffix like "Jack Flaherty (#656427)")
            repertoire = {}
            if player_name:
                if player_name in model.repertoire:
                    repertoire = model.repertoire[player_name]
                else:
                    raw_name = player_name.split(' (')[0]
                    repertoire = model.repertoire.get(raw_name, {})
                    
            return jsonify({
                'player': player_name,
                'type': player_type,
                'stats': stats,
                'repertoire': repertoire
            }), 200
        else:
            return jsonify({'error': 'Invalid type. Use "batter" or "pitcher"'}), 400
        
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import os
    port = 5000 if os.path.exists('/.dockerenv') or os.environ.get('FLASK_ENV') == 'production' else 5001
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)