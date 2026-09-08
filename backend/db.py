"""
Database module for Mapping Heat
Manages SQLite connection and pitch data loading
"""
import sqlite3
import os
import glob
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = '/app/instance/mapping_heat.sqlite'
if not os.path.exists('/app'):
    DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'instance', 'mapping_heat.sqlite'))

SCHEMA_PATH = '/app/schema.sql'
if not os.path.exists('/app'):
    SCHEMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schema.sql'))

ARTIFACTS_DIR = '/app/artifacts'
if not os.path.exists('/app'):
    ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts'))


import unicodedata

def normalize_name(text):
    if not text:
        return ""
    # Strip accents
    normalized = ''.join(c for c in unicodedata.normalize('NFD', text)
                        if unicodedata.category(c) != 'Mn')
    # Strip dots to handle initials (e.g. A.J. Blubaugh -> AJ Blubaugh)
    normalized = normalized.replace('.', '')
    # Split, lowercase, sort alphabetically, and join
    words = sorted([w.strip().lower() for w in normalized.replace(',', ' ').split() if w.strip()])
    return ''.join(words)



def get_db_connection():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.create_function("NORMALIZE", 1, normalize_name)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with schema and load pitch data from CSV."""
    logger.info(f"Initializing database at {DB_PATH}")
    conn = get_db_connection()
    
    try:
        with open(SCHEMA_PATH, 'r') as f:
            conn.executescript(f.read())
        logger.info("Database schema created")
        
        csv_pattern = os.path.join(ARTIFACTS_DIR, 'pitching_data_*.csv')
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.warning(f"No pitch data CSV found at {csv_pattern}")
            logger.warning("Database will be empty. Run pipeline.py to generate data.")
            conn.close()
            return
        
        # Use the CSV with the largest size (most comprehensive dataset)
        csv_file = max(csv_files, key=os.path.getsize)
        logger.info(f"Loading pitch data from {csv_file}")
        
        df = pd.read_csv(csv_file, low_memory=False)
        logger.info(f"Loaded {len(df):,} pitches from CSV")
        
        required_cols = [
            'pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z',
            'player_name', 'batter', 'pitcher', 'events', 'description', 'zone',
            'stand', 'p_throws', 'type', 'balls', 'strikes', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning',
            'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension',
            'spin_axis', 'arm_angle', 'n_thruorder_pitcher'
        ]
        
        available_cols = [col for col in required_cols if col in df.columns]
        df_filtered = df[available_cols]
        
        logger.info("Writing pitches to database")
        df_filtered.to_sql('pitching_data', conn, if_exists='replace', index=False)
        
        count = conn.execute("SELECT COUNT(*) FROM pitching_data").fetchone()[0]
        logger.info(f"Database loaded with {count:,} pitches")
        
        pitcher_count = conn.execute("SELECT COUNT(DISTINCT player_name) FROM pitching_data").fetchone()[0]
        logger.info(f"Found {pitcher_count} unique pitchers")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        conn.close()


def query_pitcher_pitches(pitcher_name: str):
    """Query all pitches for a specific pitcher."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM pitching_data WHERE NORMALIZE(player_name) = NORMALIZE(?) ORDER BY game_date DESC",
            (pitcher_name,)
        )
        pitches = [dict(row) for row in cursor.fetchall()]
        return pitches
    finally:
        conn.close()


def query_pitcher_names():
    """Get list of all pitcher names."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT DISTINCT player_name FROM pitching_data ORDER BY player_name ASC"
        )
        names = [row[0] for row in cursor.fetchall()]
        return names
    finally:
        conn.close()


def query_batter_stances():
    """Get mapping of batter names to their batting stance (R/L)."""
    conn = get_db_connection()
    try:
        # Get the most common stance for each batter from the database
        cursor = conn.execute("""
            SELECT batter_name, stand, COUNT(*) as cnt
            FROM pitching_data
            WHERE batter_name IS NOT NULL AND stand IS NOT NULL
            GROUP BY batter_name, stand
            ORDER BY batter_name, cnt DESC
        """)

        # Build a dict with each batter's most common stance
        stances = {}
        for row in cursor.fetchall():
            batter_name = row[0]
            stance = row[1]
            if batter_name not in stances:
                stances[batter_name] = stance

        return stances
    finally:
        conn.close()


def query_pitcher_pitches_by_id(pitcher_id: int):
    """Query all pitches for a specific pitcher by their MLBAM ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM pitching_data WHERE CAST(pitcher AS INTEGER) = ? ORDER BY game_date DESC",
            (pitcher_id,)
        )
        pitches = [dict(row) for row in cursor.fetchall()]
        return pitches
    finally:
        conn.close()