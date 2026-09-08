"""
Stats Blueprint - Legacy endpoints for pitch visualization
Provides historical pitch data from database
"""
from flask import Blueprint, request, jsonify
import logging
from db import query_pitcher_pitches, query_pitcher_names, query_pitcher_pitches_by_id

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
    pitcher_id = request.args.get('id')
    pitcher_name = request.args.get('name')
    
    try:
        if pitcher_id:
            pitches = query_pitcher_pitches_by_id(int(pitcher_id))
            logger.info(f"Returned {len(pitches)} pitches for ID {pitcher_id}")
        elif pitcher_name:
            pitches = query_pitcher_pitches(pitcher_name)
            logger.info(f"Returned {len(pitches)} pitches for {pitcher_name}")
        else:
            return jsonify({'error': 'Missing id or name parameter'}), 400
            
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


@bp.route('/matchup-at-bats', methods=['GET'])
def matchup_at_bats():
    """Returns unique at-bats for a specific pitcher-batter matchup."""
    pitcher_id = request.args.get('pitcher_id')
    batter_id = request.args.get('batter_id')
    
    if not pitcher_id or not batter_id:
        return jsonify({'error': 'Missing pitcher_id or batter_id parameter'}), 400
        
    try:
        from db import get_db_connection
        conn = get_db_connection()
        cursor = conn.execute("""
            SELECT game_date, inning, outs_when_up, MAX(events) as final_event, COUNT(*) as pitch_count
            FROM pitching_data
            WHERE pitcher = ? AND batter = ?
            GROUP BY game_date, inning, outs_when_up
            ORDER BY game_date DESC, inning ASC
        """, (float(pitcher_id), float(batter_id)))
        
        at_bats = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            at_bats.append({
                'game_date': row_dict['game_date'],
                'inning': int(row_dict['inning']),
                'outs_when_up': int(row_dict['outs_when_up']),
                'pitch_count': row_dict['pitch_count'],
                'final_event': row_dict['final_event'] or 'In-progress/Other'
            })
            
        conn.close()
        return jsonify(at_bats), 200
    except Exception as e:
        logger.error(f"Error fetching matchup at-bats: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/at-bat-pitches', methods=['GET'])
def at_bat_pitches():
    """Returns chronological pitches for a specific matchup at-bat."""
    pitcher_id = request.args.get('pitcher_id')
    batter_id = request.args.get('batter_id')
    game_date = request.args.get('game_date')
    inning = request.args.get('inning')
    outs_when_up = request.args.get('outs_when_up')
    
    if not all([pitcher_id, batter_id, game_date, inning, outs_when_up]):
        return jsonify({'error': 'Missing required query parameters'}), 400
        
    try:
        from db import get_db_connection
        conn = get_db_connection()
        cursor = conn.execute("""
            SELECT *
            FROM pitching_data
            WHERE pitcher = ? AND batter = ? AND game_date = ? AND inning = ? AND outs_when_up = ?
            ORDER BY rowid DESC
        """, (float(pitcher_id), float(batter_id), game_date, float(inning), float(outs_when_up)))
        
        pitches = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(pitches), 200
    except Exception as e:
        logger.error(f"Error fetching at-bat pitches: {e}")
        return jsonify({'error': str(e)}), 500