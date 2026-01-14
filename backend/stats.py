"""
Stats Blueprint - Legacy endpoints for pitch visualization
Provides historical pitch data from database
"""
from flask import Blueprint, request, jsonify
import logging
from db import query_pitcher_pitches, query_pitcher_names

logger = logging.getLogger(__name__)

bp = Blueprint('stats', __name__, url_prefix='/stats')


@bp.route('/pitchers/names', methods=['GET'])
def pitcher_names():
    """Returns list of unique pitcher names from database."""
    try:
        names = query_pitcher_names()
        logger.info(f"Returned {len(names)} pitcher names")
        return jsonify(names), 200
    except Exception as e:
        logger.error(f"Error fetching pitcher names: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/pitcher', methods=['GET'])
def pitcher():
    """Returns all historical pitch data for a specific pitcher."""
    pitcher_name = request.args.get('name')
    
    if not pitcher_name:
        return jsonify({'error': 'Missing name parameter'}), 400
    
    try:
        pitches = query_pitcher_pitches(pitcher_name)
        logger.info(f"Returned {len(pitches)} pitches for {pitcher_name}")
        return jsonify(pitches), 200
    except Exception as e:
        logger.error(f"Error fetching pitcher data: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/predict', methods=['POST'])
def predict_zones():
    """
    Legacy endpoint - predicts probabilities for all zones.
    This is for backward compatibility with the original frontend.
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 415
    
    data = request.get_json()
    
    # This will be handled by the main predict endpoint
    # For now, return a simple response
    logger.info("Legacy /stats/predict called - redirecting to new endpoint")
    return jsonify({'message': 'Use /predict endpoint instead'}), 200