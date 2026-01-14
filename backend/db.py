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
SCHEMA_PATH = '/app/schema.sql'
ARTIFACTS_DIR = '/app/artifacts'


def get_db_connection():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with schema and load pitch data from CSV."""
    logger.info("=" * 60)
    logger.info("Initializing Database")
    logger.info("=" * 60)
    
    # Create schema
    logger.info(f"Creating database at: {DB_PATH}")
    conn = get_db_connection()
    
    try:
        # Execute schema
        logger.info(f"Loading schema from: {SCHEMA_PATH}")
        with open(SCHEMA_PATH, 'r') as f:
            conn.executescript(f.read())
        logger.info("✓ Database schema created")
        
        # Find pitch data CSV
        csv_pattern = os.path.join(ARTIFACTS_DIR, 'pitching_data_*.csv')
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.warning(f"⚠ No pitch data CSV found at {csv_pattern}")
            logger.warning("  Database will be empty. Run pipeline.py to generate data.")
            conn.close()
            return
        
        # Use most recent CSV
        csv_file = sorted(csv_files)[-1]
        logger.info(f"Loading pitch data from: {csv_file}")
        
        # Load CSV into database
        df = pd.read_csv(csv_file, low_memory=False)
        logger.info(f"  Loaded {len(df):,} pitches from CSV")
        
        # Select only the columns we need (matching schema)
        required_cols = [
            'pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z',
            'player_name', 'batter', 'pitcher', 'events', 'description', 'zone',
            'stand', 'p_throws', 'type', 'balls', 'strikes', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning',
            'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension',
            'spin_axis', 'arm_angle', 'n_thruorder_pitcher'
        ]
        
        # Keep only available columns
        available_cols = [col for col in required_cols if col in df.columns]
        df_filtered = df[available_cols]
        
        logger.info(f"  Writing to database...")
        df_filtered.to_sql('pitching_data', conn, if_exists='replace', index=False)
        
        # Verify
        count = conn.execute("SELECT COUNT(*) FROM pitching_data").fetchone()[0]
        logger.info(f"✓ Database loaded with {count:,} pitches")
        
        # Get pitcher count
        pitcher_count = conn.execute("SELECT COUNT(DISTINCT player_name) FROM pitching_data").fetchone()[0]
        logger.info(f"✓ Found {pitcher_count} unique pitchers")
        
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    finally:
        conn.close()
    
    logger.info("=" * 60)
    logger.info("✓ Database Ready!")
    logger.info("=" * 60)


def query_pitcher_pitches(pitcher_name: str):
    """Query all pitches for a specific pitcher."""
    conn = get_db_connection()
    try:
        # Convert "First Last" to "Last, First" format for database query
        # Database stores names as "Last, First" but UI uses "First Last"
        name_parts = pitcher_name.strip().split(' ', 1)
        if len(name_parts) == 2:
            db_name = f"{name_parts[1]}, {name_parts[0]}"
        else:
            db_name = pitcher_name

        cursor = conn.execute(
            "SELECT * FROM pitching_data WHERE player_name = ? ORDER BY game_date DESC",
            (db_name,)
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