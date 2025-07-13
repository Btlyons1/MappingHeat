from flask import Blueprint, request, jsonify, Response
from typing import Dict, List, Optional
from .db import get_db
from .model import predict_hit_prob

bp = Blueprint('stats', __name__, url_prefix='/stats')

@bp.route('/pitchers/names', methods=['GET'])
def pitcher_names() -> Response:
    """Returns a list of unique pitcher names from the database."""
    name_query = "SELECT DISTINCT player_name FROM pitching_data ORDER BY player_name ASC"
    try:
        names = [row[0] for row in get_db().execute(name_query).fetchall()]
        return jsonify(names)
    except Exception as e:
        print(f"❌ DB error in /pitchers/names: {e}")
        return jsonify({"error": "Database error"}), 500

@bp.route('/pitcher', methods=['GET'])
def pitcher() -> Response:
    """Returns all historical pitch data for a specific pitcher."""
    name: Optional[str] = request.args.get('name')
    if not name:
        return jsonify({"error": "Missing 'name' parameter"}), 400
    
    pitch_query = "SELECT * FROM pitching_data WHERE player_name = ?"
    try:
        pitcher_data: List[Dict] = [dict(row) for row in get_db().execute(pitch_query, (name,)).fetchall()]
        return jsonify(pitcher_data)
    except Exception as e:
        print(f"❌ DB error in /pitcher endpoint: {e}")
        return jsonify({"error": "Database error"}), 500

@bp.route('/predict', methods=['POST'])
def predict_zones_dynamically() -> Response:
    """
    Receives a full pitch context from the frontend sliders via a POST request,
    iterates through all zones, and returns a dictionary of per-zone probabilities.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    base_pitch_context: Optional[Dict] = request.get_json()
    if not isinstance(base_pitch_context, dict):
         return jsonify({"error": "Invalid JSON payload"}), 400

    results: Dict[str, Optional[float]] = {}
    
    zone_coords = {
        '1': (-0.55, 3.2), '2': (0.0, 3.2), '3': (0.55, 3.2),
        '4': (-0.55, 2.5), '5': (0.0, 2.5), '6': (0.55, 2.5),
        '7': (-0.55, 1.8), '8': (0.0, 1.8), '9': (0.55, 1.8),
        '11': (-1.2, 3.5), '12': (1.2, 3.5),
        '13': (-1.2, 1.5), '14': (1.2, 1.5)
    }

    print(f"Generating dynamic per-zone predictions for: {base_pitch_context.get('player_name')}")
    for zone, coords in zone_coords.items():
        current_pitch_context = base_pitch_context.copy()
        
        current_pitch_context['zone'] = zone
        current_pitch_context['plate_x'] = coords[0]
        current_pitch_context['plate_z'] = coords[1]
        
        probability = predict_hit_prob(current_pitch_context)
        results[zone] = probability if probability is not None else 0.0
    
    print(f"✅ Returning dynamic probabilities for all zones.")
    return jsonify(results)