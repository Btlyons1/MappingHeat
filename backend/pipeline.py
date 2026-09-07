"""
Mapping Heat Data Pipeline
Fetches Statcast and Fangraphs data, trains LightGBM model, saves artifacts.
"""
import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from pybaseball import statcast, batting_stats, pitching_stats, chadwick_register
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_network_connection():
    """Test if we can connect to required services."""
    import requests
    
    logger.info("Testing network connectivity...")
    
    try:
        # Test MLB Statcast API
        response = requests.get('https://baseballsavant.mlb.com', timeout=10)
        logger.info("MLB Statcast API accessible")
        return True
    except requests.exceptions.SSLError as e:
        logger.error("SSL Error: urllib3 compatibility issue detected")
        logger.error(f"{e}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return False


class MappingHeatPipeline:
    """Production pipeline for Mapping Heat model training and artifact generation."""
    
    def __init__(self, start_date: str, end_date: str, output_dir: str = 'artifacts', cache_dir: str = 'cache'):
        """
        Initialize pipeline.

        Args:
            start_date: Start date for Statcast data (YYYY-MM-DD)
            end_date: End date for Statcast data (YYYY-MM-DD)
            output_dir: Directory to save artifacts
            cache_dir: Directory to cache intermediate data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.pitch_data = None
        self.batter_stats = None
        self.pitcher_stats = None
        self.league_averages = {}
        self.id_mapping = None
        self.label_encoders = {}
        self.model = None
        
        # Columns to keep from Statcast
        self.cols_to_keep = [
            'player_name', 'pitch_type', 'zone', 'stand', 'p_throws',
            'release_speed', 'balls', 'strikes', 'pfx_x', 'pfx_z',
            'release_spin_rate', 'release_extension', 'plate_x', 'plate_z',
            'effective_speed', 'spin_axis', 'release_pos_x', 'release_pos_z',
            'arm_angle', 'on_1b', 'on_2b', 'on_3b', 'outs_when_up', 'inning',
            'sz_top', 'sz_bot', 'n_thruorder_pitcher', 'events', 'game_date',
            'type', 'description',  # type: S/B/X for Strike/Ball/InPlay; description: detailed outcome
            'game_pk', 'at_bat_number', 'pitch_number'  # For pitch sequencing and tunneling
        ]
        
        # Additional ID columns needed for merging
        self.merge_cols = ['batter', 'pitcher']
    
    def fetch_season_stats(self, years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch season-level batting and pitching stats from Fangraphs for multiple years.

        Args:
            years: List of season years

        Returns:
            Tuple of (batter_stats, pitcher_stats) DataFrames
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f'season_stats_{min(years)}_{max(years)}.pkl')
        fallback_cache = os.path.join(self.cache_dir, 'season_stats_2023_2025.pkl')
        
        selected_cache = None
        if os.path.exists(cache_file):
            selected_cache = cache_file
        elif os.path.exists(fallback_cache):
            logger.info(f"Standard cache {cache_file} not found, but fallback cache {fallback_cache} exists. Using it.")
            selected_cache = fallback_cache
            
        if selected_cache:
            logger.info(f"Loading cached season stats from {selected_cache}")
            with open(selected_cache, 'rb') as f:
                cached = pickle.load(f)
            batting = cached['batting']
            pitching = cached['pitching']
            # Filter by the requested years
            batting = batting[batting['year'].isin(years)].reset_index(drop=True)
            pitching = pitching[pitching['year'].isin(years)].reset_index(drop=True)
            return batting, pitching

        all_batting = []
        all_pitching = []

        for year in years:
            logger.info(f"Fetching season stats for {year}...")
            
            try:
                # Fetch batting stats
                logger.info(f"Downloading batting data for {year}")
                batting = batting_stats(year, qual=150)
                batting['year'] = year
                logger.info(f"Fetched {len(batting)} batters for {year}")
                all_batting.append(batting)
            except Exception as e:
                logger.error(f"Failed to fetch batting stats for {year}: {e}")
                raise
            
            try:
                # Fetch pitching stats
                logger.info(f"Downloading pitching data for {year}")
                pitching = pitching_stats(year, qual=20)
                pitching['year'] = year
                logger.info(f"Fetched {len(pitching)} pitchers for {year}")
                all_pitching.append(pitching)
            except Exception as e:
                logger.error(f"Failed to fetch pitching stats for {year}: {e}")
                raise
        
        # Combine all years
        combined_batting = pd.concat(all_batting, ignore_index=True)
        combined_pitching = pd.concat(all_pitching, ignore_index=True)

        logger.info(f"Total batters across all years: {len(combined_batting)}")
        logger.info(f"Total pitchers across all years: {len(combined_pitching)}")

        # Cache the results
        logger.info(f"Caching season stats to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({'batting': combined_batting, 'pitching': combined_pitching}, f)

        return combined_batting, combined_pitching
    
    def calculate_league_averages(self, batting: pd.DataFrame, pitching: pd.DataFrame) -> Dict:
        """
        Calculate league-wide averages dynamically from fetched data.
        
        Args:
            batting: Batter stats DataFrame
            pitching: Pitcher stats DataFrame
            
        Returns:
            Dictionary of league averages
        """
        logger.info("Calculating league averages")
        
        # Batting averages
        batter_keys = ['wOBA', 'OBP', 'SLG', 'ISO', 'BABIP', 'BB%', 'K%']
        pitcher_keys = ['FIP', 'ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'BABIP']
        
        averages = {}
        
        # Calculate batting averages
        for key in batter_keys:
            if key in batting.columns:
                averages[f'batter_{key}'] = float(batting[key].mean())
        
        # Calculate pitching averages
        for key in pitcher_keys:
            if key in pitching.columns:
                averages[f'pitcher_{key}'] = float(pitching[key].mean())
        
        logger.info(f"Calculated {len(averages)} league averages")
        return averages
    
    def fetch_id_mapping(self) -> pd.DataFrame:
        """
        Fetch player ID mapping from Chadwick to map MLBAM <-> Fangraphs IDs.

        Returns:
            DataFrame with ID mappings
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, 'id_mapping.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached ID mapping from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("Fetching player ID mappings")

        # Get the full Chadwick register database with all player ID mappings
        id_map = chadwick_register()

        logger.info(f"Fetched mappings for {len(id_map)} players")
        logger.info(f"Available ID columns: {[col for col in id_map.columns if 'key_' in col]}")

        # Cache the results
        logger.info(f"Caching ID mapping to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(id_map, f)

        return id_map
    
    def fetch_statcast_bulk(self) -> pd.DataFrame:
        """
        Fetch Statcast pitch data in bulk by date range.
        Uses chunking to avoid MLB server timeout/parsing errors.

        Returns:
            DataFrame of pitch-level data
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f'statcast_{self.start_date}_to_{self.end_date}.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached Statcast data from {cache_file}")
            with open(cache_file, 'rb') as f:
                pitch_data = pickle.load(f)
            logger.info(f"Loaded {len(pitch_data):,} cached pitches")
            return pitch_data

        logger.info(f"Fetching Statcast data from {self.start_date} to {self.end_date}")

        from datetime import datetime, timedelta

        try:
            # Parse dates
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            # Fetch in monthly chunks to avoid parser errors
            all_data = []
            current = start
            chunk_num = 1
            
            while current <= end:
                # Calculate chunk end (1 month or end date, whichever is earlier)
                chunk_end = min(
                    current + timedelta(days=30),
                    end
                )
                
                chunk_start_str = current.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                
                logger.info(f"Chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}")
                
                try:
                    chunk_data = statcast(start_dt=chunk_start_str, end_dt=chunk_end_str)
                    
                    if chunk_data is not None and len(chunk_data) > 0:
                        all_data.append(chunk_data)
                        logger.info(f"Fetched {len(chunk_data):,} pitches")
                    else:
                        logger.warning("No data returned for this chunk")
                        
                except Exception as e:
                    logger.error(f"Chunk failed: {e}")
                    logger.warning("Continuing with next chunk")
                
                # Move to next chunk
                current = chunk_end + timedelta(days=1)
                chunk_num += 1
            
            if not all_data:
                raise ValueError("No pitch data fetched from any chunk")
            
            # Combine all chunks
            pitch_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total fetched: {len(pitch_data):,} pitches from {len(all_data)} chunks")
            
            # Keep only necessary columns (including IDs for merging)
            available_cols = [col for col in self.cols_to_keep + self.merge_cols if col in pitch_data.columns]
            missing_cols = [col for col in self.cols_to_keep if col not in pitch_data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            
            pitch_data = pitch_data[available_cols].copy()
            logger.info(f"Kept {len(available_cols)} relevant columns")

            # Cache the results
            logger.info(f"Caching Statcast data to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(pitch_data, f)

            return pitch_data
            
        except Exception as e:
            logger.error(f"  ✗ Failed to fetch Statcast data: {e}")
            logger.error("  Possible causes:")
            logger.error("    - Network connection issue")
            logger.error("    - MLB Statcast server down")
            logger.error("    - SSL/urllib3 compatibility issue")
            logger.error("    - Invalid date range")
            logger.error("\n  TRY:")
            logger.error("    - Clear cache: rm -rf ~/.pybaseball")
            logger.error("    - Reduce date range")
            logger.error("    - Try again later")
            raise
    
    def merge_context(self, pitches: pd.DataFrame, batting: pd.DataFrame, 
                      pitching: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
        """
        Merge season stats onto pitch data using proper ID mapping.
        Handles multi-year data by matching on both player ID and year.
        
        Args:
            pitches: Pitch-level Statcast data
            batting: Batter season stats
            pitching: Pitcher season stats
            id_map: Player ID mapping
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging context onto pitch data")
        
        # Extract year from game_date in pitches
        if 'game_date' in pitches.columns:
            pitches['year'] = pd.to_datetime(pitches['game_date']).dt.year
        
        logger.info(f"Pitch data years: {pitches['year'].unique()}")
        logger.info(f"Batting data years: {batting['year'].unique()}")
        logger.info(f"Pitching data years: {pitching['year'].unique()}")
        
        # Prepare ID mapping for batters (MLBAM -> Fangraphs)
        batter_id_map = id_map[['key_mlbam', 'key_fangraphs']].copy()
        batter_id_map.columns = ['batter', 'batter_fg_id']
        batter_id_map = batter_id_map.dropna(subset=['batter', 'batter_fg_id'])
        batter_id_map['batter'] = batter_id_map['batter'].astype(int)
        batter_id_map['batter_fg_id'] = batter_id_map['batter_fg_id'].astype(int)
        
        logger.info(f"Batter ID mappings: {len(batter_id_map)}")
        
        # Prepare ID mapping for pitchers (MLBAM -> Fangraphs)
        pitcher_id_map = id_map[['key_mlbam', 'key_fangraphs']].copy()
        pitcher_id_map.columns = ['pitcher', 'pitcher_fg_id']
        pitcher_id_map = pitcher_id_map.dropna(subset=['pitcher', 'pitcher_fg_id'])
        pitcher_id_map['pitcher'] = pitcher_id_map['pitcher'].astype(int)
        pitcher_id_map['pitcher_fg_id'] = pitcher_id_map['pitcher_fg_id'].astype(int)
        
        logger.info(f"Pitcher ID mappings: {len(pitcher_id_map)}")
        
        # Add Fangraphs IDs to pitches
        logger.info("Adding Fangraphs IDs to pitch data...")
        pitches = pitches.merge(batter_id_map, on='batter', how='left')
        pitches = pitches.merge(pitcher_id_map, on='pitcher', how='left')
        
        # Check how many pitches have FG IDs
        batter_with_fg = pitches['batter_fg_id'].notna().sum()
        pitcher_with_fg = pitches['pitcher_fg_id'].notna().sum()
        logger.info(f"Pitches with batter FG ID: {batter_with_fg:,} / {len(pitches):,} ({batter_with_fg/len(pitches)*100:.1f}%)")
        logger.info(f"Pitches with pitcher FG ID: {pitcher_with_fg:,} / {len(pitches):,} ({pitcher_with_fg/len(pitches)*100:.1f}%)")
        
        # Prepare batting stats with fg ID
        batting_merge = batting.rename(columns={'IDfg': 'batter_fg_id'})
        batting_merge['batter_fg_id'] = batting_merge['batter_fg_id'].astype(int)
        
        # Keep only the rate stats we want
        batter_cols_to_keep = ['batter_fg_id', 'year', 'wOBA', 'OBP', 'SLG', 'BB%', 'K%', 'ISO', 'BABIP', 'HardHit%', 'Barrel%', 'Contact%', 'O-Swing%']
        available_batter_cols = [col for col in batter_cols_to_keep if col in batting_merge.columns or col == 'batter_fg_id' or col == 'year']
        batting_merge = batting_merge[available_batter_cols]
        batting_merge = batting_merge.add_prefix('batter_')
        batting_merge.rename(columns={'batter_batter_fg_id': 'batter_fg_id', 'batter_year': 'year'}, inplace=True)
        
        logger.info(f"Batting stats columns: {list(batting_merge.columns)}")
        logger.info(f"Batting stats shape: {batting_merge.shape}")
        
        # Prepare pitching stats with fg ID
        pitching_merge = pitching.rename(columns={'IDfg': 'pitcher_fg_id'})
        pitching_merge['pitcher_fg_id'] = pitching_merge['pitcher_fg_id'].astype(int)
        
        # Keep only the rate stats we want
        pitcher_cols_to_keep = ['pitcher_fg_id', 'year', 'FIP', 'ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'xFIP', 'xERA', 'SwStr%']
        available_pitcher_cols = [col for col in pitcher_cols_to_keep if col in pitching_merge.columns or col == 'pitcher_fg_id' or col == 'year']
        pitching_merge = pitching_merge[available_pitcher_cols]
        pitching_merge = pitching_merge.add_prefix('pitcher_')
        pitching_merge.rename(columns={'pitcher_pitcher_fg_id': 'pitcher_fg_id', 'pitcher_year': 'year'}, inplace=True)
        
        logger.info(f"Pitching stats columns: {list(pitching_merge.columns)}")
        logger.info(f"Pitching stats shape: {pitching_merge.shape}")
        
        # Merge batting stats (on both player ID and year)
        logger.info("Merging batting stats...")
        before_merge = len(pitches)
        pitches = pitches.merge(batting_merge, on=['batter_fg_id', 'year'], how='left')
        logger.info(f"After batting merge: {len(pitches)} rows (should be same as before: {before_merge})")
        
        # Check how many got stats
        woba_populated = pitches['batter_wOBA'].notna().sum() if 'batter_wOBA' in pitches.columns else 0
        logger.info(f"Pitches with batter_wOBA: {woba_populated:,} / {len(pitches):,} ({woba_populated/len(pitches)*100:.1f}%)")
        
        # Merge pitching stats (on both player ID and year)
        logger.info("Merging pitching stats...")
        pitches = pitches.merge(pitching_merge, on=['pitcher_fg_id', 'year'], how='left')
        
        # Check how many got stats
        fip_populated = pitches['pitcher_FIP'].notna().sum() if 'pitcher_FIP' in pitches.columns else 0
        logger.info(f"Pitches with pitcher_FIP: {fip_populated:,} / {len(pitches):,} ({fip_populated/len(pitches)*100:.1f}%)")
        
        logger.info(f"Final merged data shape: {pitches.shape}")
        logger.info(f"Merged data columns: {list(pitches.columns)}")
        
        return pitches
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features and prepare for modeling.

        Args:
            df: Merged DataFrame

        Returns:
            Feature-engineered DataFrame
        """
        logger.info("Engineering features")

        # Sort chronologically to calculate sequence features correctly
        logger.info("Sorting pitches chronologically to build sequence features...")
        sort_cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
        # Check if all sort columns are present
        present_sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(by=present_sort_cols).reset_index(drop=True)

        # Shift features within each plate appearance (game_pk + at_bat_number)
        logger.info("Extracting sequential pitch context features...")
        gp = df.groupby(['game_pk', 'at_bat_number'])
        
        # 1. Previous pitch type (categorical)
        df['prev_pitch_type'] = gp['pitch_type'].shift(1).fillna('None')
        
        # 2. Previous release speed
        prev_release_speed = gp['release_speed'].shift(1)
        # Pitch speed diff: current_speed - prev_speed
        df['prev_pitch_speed_diff'] = df['release_speed'] - prev_release_speed
        df['prev_pitch_speed_diff'] = df['prev_pitch_speed_diff'].fillna(0.0)
        
        # 3. Previous pitch coordinates (plate_x, plate_z)
        prev_plate_x = gp['plate_x'].shift(1)
        prev_plate_z = gp['plate_z'].shift(1)
        # Tunneling delta: distance between current and previous location in feet
        df['prev_pitch_location_dist'] = np.sqrt(
            (df['plate_x'] - prev_plate_x)**2 + 
            (df['plate_z'] - prev_plate_z)**2
        )
        df['prev_pitch_location_dist'] = df['prev_pitch_location_dist'].fillna(0.0)
        
        # 4. Pitch number: ensure it is numeric and filled
        if 'pitch_number' in df.columns:
            df['pitch_number'] = pd.to_numeric(df['pitch_number'], errors='coerce').fillna(1).astype(int)
        else:
            # Fallback calculation if not in columns
            df['pitch_number'] = gp.cumcount() + 1

        # Create binary target: Hit (1) vs Out (0)
        hit_events = ['single', 'double', 'triple', 'home_run']
        out_events = ['field_out', 'strikeout', 'force_out', 'grounded_into_double_play',
                      'double_play', 'fielders_choice_out', 'strikeout_double_play']

        df['target'] = df['events'].apply(
            lambda x: 1 if x in hit_events else (0 if x in out_events else np.nan)
        )

        # Drop rows without valid outcomes
        df = df.dropna(subset=['target'])

        # Convert baserunner columns to binary
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if base in df.columns:
                df[base] = df[base].notna().astype(int)

        # NEW: Add advanced baseball context features
        logger.info("Adding advanced baseball context features...")
        if 'stand' in df.columns and 'p_throws' in df.columns:
            df['is_platoon_advantage'] = (df['stand'] != df['p_throws']).astype(int)
            
        if 'effective_speed' in df.columns and 'release_speed' in df.columns:
            df['perceived_speed_diff'] = df['effective_speed'] - df['release_speed']
            
        if 'release_speed' in df.columns and 'plate_x' in df.columns:
            df['speed_x_location'] = df['release_speed'] * df['plate_x'].abs()
        if 'release_speed' in df.columns and 'plate_z' in df.columns:
            df['speed_x_height'] = df['release_speed'] * df['plate_z']
            
        # Fastball velocity differential
        pitcher_id_col = 'pitcher_fg_id' if 'pitcher_fg_id' in df.columns else 'pitcher'
        fb_pitches = df[df['pitch_type'].isin(['FF', 'SI'])]
        if len(fb_pitches) > 0 and 'release_speed' in df.columns:
            fb_speeds = fb_pitches.groupby(pitcher_id_col)['release_speed'].mean()
            df['pitcher_avg_fb_speed'] = df[pitcher_id_col].map(fb_speeds)
            df['pitcher_avg_fb_speed'] = df['pitcher_avg_fb_speed'].fillna(93.0)
            df['velocity_differential'] = df['pitcher_avg_fb_speed'] - df['release_speed']
        else:
            df['velocity_differential'] = 0.0

        # NEW: Add count context features (Week 1 improvement)
        logger.info("Adding count context features...")
        if 'balls' in df.columns and 'strikes' in df.columns:
            # Hitter's count: batter has advantage (2-0, 3-0, 3-1)
            df['is_hitters_count'] = ((df['balls'] >= 2) & (df['strikes'] <= 1)).astype(int)

            # Pitcher's count: pitcher has advantage (0-2, 1-2)
            df['is_pitchers_count'] = ((df['strikes'] == 2) & (df['balls'] <= 1)).astype(int)

            # Count leverage: normalized -1 (pitcher advantage) to 1 (hitter advantage)
            df['count_leverage'] = (df['balls'] - df['strikes']) / 3.0

            # Two-strike pressure
            df['two_strike_count'] = (df['strikes'] == 2).astype(int)

            logger.info(f"  Added 4 count context features")
            logger.info(f"  Hitter's count: {df['is_hitters_count'].sum():,} pitches ({df['is_hitters_count'].mean()*100:.1f}%)")
            logger.info(f"  Pitcher's count: {df['is_pitchers_count'].sum():,} pitches ({df['is_pitchers_count'].mean()*100:.1f}%)")

        # Fill missing numeric values with median (fix pandas warning)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Encode categorical variables
        categorical_cols = ['pitch_type', 'zone', 'stand', 'p_throws', 'prev_pitch_type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('Unknown')
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

        return df
    
    def train_model(self, df: pd.DataFrame) -> lgb.LGBMClassifier:
        """
        Train LightGBM classification model.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            Trained model
        """
        logger.info("Training LightGBM model")
        
        # Define columns to EXCLUDE (metadata and identifiers)
        # Exact match columns (must match exactly)
        exclude_exact = [
            'player_name', 'events', 'game_date', 'target', 'batter', 'pitcher',
            'batter_fg_id', 'pitcher_fg_id', 'year',
            'Name', 'Team', 'PlayerID', 'playerid', 'Season',
            'Dollars', 'AuctionVal', 'RAR', 'WAR', 'Age Rng',
            # Sequence/tunneling metadata and intermediate variables
            'game_pk', 'at_bat_number', 'prev_release_speed', 'prev_plate_x', 'prev_plate_z', 'type', 'description'
        ]

        # Get all columns
        all_cols = df.columns.tolist()

        # Remove excluded columns
        feature_cols = []
        for col in all_cols:
            # Skip if in exact exclude list
            if col in exclude_exact:
                continue

            # Skip counting stats but KEEP rate stats (those with %)
            # Counting stats end with the stat name (e.g., batter_HR, batter_BB, batter_SO)
            # Rate stats have % or / in them (e.g., batter_BB%, pitcher_K/9)
            if '%' not in col and '/' not in col:
                # This might be a counting stat - check if it ends with a counting stat suffix
                counting_suffixes = ['_G', '_PA', '_AB', '_H', '_1B', '_2B', '_3B', '_HR',
                                   '_R', '_RBI', '_BB', '_IBB', '_SO', '_HBP', '_SF', '_SH',
                                   '_GDP', '_SB', '_CS', '_W', '_L', '_SV', '_GS', '_IP',
                                   '_TBF', '_ER', '_WP', '_BK', '_Age']
                if any(col.endswith(suffix) for suffix in counting_suffixes):
                    continue
            
            # Skip non-numeric columns (include pandas nullable numeric types)
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            feature_cols.append(col)
        
        logger.info(f"Selected {len(feature_cols)} numeric features for training")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Excluded {len(all_cols) - len(feature_cols) - 1} non-numeric/metadata columns")
        
        X = df[feature_cols]
        y = df['target']
        
        # Double-check for any remaining non-numeric columns
        non_numeric = X.select_dtypes(include=['object']).columns.tolist()
        if non_numeric:
            logger.warning(f"Found non-numeric columns in features: {non_numeric}")
            logger.info("Removing these columns...")
            X = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'bool', 'int8', 'uint8'])
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train model with tuned hyperparameters for new baseball features
        model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            min_child_samples=50,
            # Retain unweighted training to preserve probability calibration with MLB baseline hit rate
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Suppress LightGBM warnings
        )
        
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Train accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        logger.info(f"Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")

        # Save test data for model evaluation notebook
        test_data_path = os.path.join(self.output_dir, 'test_data.pkl')
        with open(test_data_path, 'wb') as f:
            pickle.dump({
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': list(X.columns)
            }, f)
        logger.info(f"Saved test data to {test_data_path} for evaluation")

        # Log feature importance (top 10) - using 'gain' for better insights
        importance_gain = model.booster_.feature_importance(importance_type='gain')
        feature_importance = sorted(
            zip(X.columns, importance_gain),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        logger.info("\nTop 10 Most Important Features (by gain):")
        for feat, importance in feature_importance:
            logger.info(f"  {feat}: {importance:.2f}")
        
        return model
    
    def calculate_pitch_profiles(self, df: pd.DataFrame) -> Dict:
        """
        Calculate average pitch characteristics by pitcher and pitch type.
        
        Args:
            df: Merged pitch data
            
        Returns:
            Dictionary of pitch profiles by pitcher and pitch type
        """
        logger.info("Calculating pitch profiles by pitcher and pitch type")
        
        pitch_cols = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 
                     'release_extension', 'spin_axis', 'arm_angle']
        
        profiles = {}
        
        # Group by player_name and pitch_type
        for (pitcher, pitch_type), group in df.groupby(['player_name', 'pitch_type']):
            if len(group) < 10:  # Minimum 10 pitches of this type
                continue
                
            pitcher_key = str(pitcher)
            if pitcher_key not in profiles:
                profiles[pitcher_key] = {}
            
            # Calculate averages for this pitch type
            pitch_profile = {}
            for col in pitch_cols:
                if col in group.columns:
                    # Calculate mean, handling NaN values
                    mean_val = group[col].mean()
                    
                    # Only add if we got a valid number (not NaN)
                    if pd.notna(mean_val):
                        pitch_profile[col] = float(mean_val)
            
            # Only save profile if we got at least some valid data
            if len(pitch_profile) > 0:
                profiles[pitcher_key][str(pitch_type)] = pitch_profile
        
        logger.info(f"Created profiles for {len(profiles)} pitchers")
        
        # Log some statistics
        total_pitch_types = sum(len(p) for p in profiles.values())
        logger.info(f"Total pitcher-pitch type combinations: {total_pitch_types}")
        
        return profiles
    
    def save_artifacts(self):
        """Save all artifacts to disk."""
        logger.info("Saving artifacts")
        
        # Save model
        model_path = os.path.join(self.output_dir, 'lgbm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders
            }, f)
        logger.info(f"Saved model to {model_path}")
        
        # Get most recent year for rosters
        most_recent_year = self.batter_stats['year'].max()
        recent_batters = self.batter_stats[self.batter_stats['year'] == most_recent_year]
        recent_pitchers = self.pitcher_stats[self.pitcher_stats['year'] == most_recent_year]
        
        logger.info(f"Using {most_recent_year} roster data for UI dropdowns")
        
        # Save batters roster (most recent year only for UI)
        batters_cols = ['Name', 'wOBA', 'OBP', 'SLG', 'BB%', 'K%', 'ISO', 'BABIP', 'HardHit%', 'Barrel%', 'Contact%', 'O-Swing%']
        batters_avail = [col for col in batters_cols if col in recent_batters.columns]
        batters = recent_batters[batters_avail].to_dict('records')
        batters_path = os.path.join(self.output_dir, 'batters.json')
        with open(batters_path, 'w') as f:
            json.dump(batters, f, indent=2)
        logger.info(f"Saved {len(batters)} batters to {batters_path}")
        
        # Calculate average fastball speed per pitcher from pitch data for UI velocity differential
        fb_speeds = {}
        if hasattr(self, 'merged_pitch_data') and 'pitcher_fg_id' in self.merged_pitch_data.columns and 'release_speed' in self.merged_pitch_data.columns:
            fb_pitches = self.merged_pitch_data[self.merged_pitch_data['pitch_type'].isin(['FF', 'SI'])]
            fb_speeds = fb_pitches.groupby('pitcher_fg_id')['release_speed'].mean().to_dict()
            
        recent_pitchers = recent_pitchers.copy()
        recent_pitchers['FB_velocity'] = recent_pitchers['IDfg'].map(fb_speeds).fillna(93.0)
        
        # Save pitchers roster (most recent year only for UI)
        pitchers_cols = ['Name', 'FIP', 'ERA', 'WHIP', 'K/9', 'BB/9', 'xFIP', 'xERA', 'SwStr%', 'FB_velocity']
        pitchers_avail = [col for col in pitchers_cols if col in recent_pitchers.columns]
        pitchers = recent_pitchers[pitchers_avail].to_dict('records')
        pitchers_path = os.path.join(self.output_dir, 'pitchers.json')
        with open(pitchers_path, 'w') as f:
            json.dump(pitchers, f, indent=2)
        logger.info(f"Saved {len(pitchers)} pitchers to {pitchers_path}")
        
        # Save league averages (calculated from all years)
        league_path = os.path.join(self.output_dir, 'league_averages.json')
        with open(league_path, 'w') as f:
            json.dump(self.league_averages, f, indent=2)
        logger.info(f"Saved league averages to {league_path}")
        
        # Save pitch profiles (from merged data before feature engineering)
        if hasattr(self, 'merged_pitch_data') and 'player_name' in self.merged_pitch_data.columns:
            pitch_profiles = self.calculate_pitch_profiles(self.merged_pitch_data)
            profiles_path = os.path.join(self.output_dir, 'pitcher_pitch_profiles.json')
            with open(profiles_path, 'w') as f:
                json.dump(pitch_profiles, f, indent=2)
            logger.info(f"Saved pitch profiles to {profiles_path}")
        else:
            logger.warning("Could not calculate pitch profiles - merged data not available")
            logger.warning("Creating empty pitch profiles file")
            profiles_path = os.path.join(self.output_dir, 'pitcher_pitch_profiles.json')
            with open(profiles_path, 'w') as f:
                json.dump({}, f, indent=2)
        
        # Calculate and save pitcher repertoire (pitch type usage percentages)
        if hasattr(self, 'merged_pitch_data') and 'player_name' in self.merged_pitch_data.columns and 'pitch_type' in self.merged_pitch_data.columns:
            logger.info("Calculating pitcher repertoires...")
            repertoire = {}
            # Group by pitcher (player_name is "Last, First")
            for pitcher_name, group in self.merged_pitch_data.groupby('player_name'):
                if len(group) < 10:  # Minimum 10 pitches to calculate repertoire
                    continue
                counts = group['pitch_type'].value_counts()
                total = len(group)
                if total == 0:
                    continue
                
                # Convert to percentages
                pitch_mix = {}
                for pt, count in counts.items():
                    pitch_mix[str(pt)] = round((count / total) * 100, 1)
                
                # Convert "Last, First" name to "First Last" to match frontend select value
                if ',' in pitcher_name:
                    parts = pitcher_name.split(',')
                    last_name = parts[0].strip()
                    first_name = parts[1].strip()
                    name_first_last = f"{first_name} {last_name}"
                else:
                    name_first_last = pitcher_name
                
                repertoire[name_first_last] = pitch_mix
                
            repertoire_path = os.path.join(self.output_dir, 'pitcher_repertoire.json')
            with open(repertoire_path, 'w') as f:
                json.dump(repertoire, f, indent=2)
            logger.info(f"Saved pitch repertoire for {len(repertoire)} pitchers to {repertoire_path}")
        else:
            logger.warning("Could not calculate pitcher repertoires - merged data not available")
            repertoire_path = os.path.join(self.output_dir, 'pitcher_repertoire.json')
            with open(repertoire_path, 'w') as f:
                json.dump({}, f, indent=2)
        
        # Save batter stances mapping from pitch data
        logger.info("Creating batter stance mapping...")
        if hasattr(self, 'merged_pitch_data') and 'stand' in self.merged_pitch_data.columns:
            # Get batter names from the batting stats
            # Map batter FG IDs to names
            batter_id_to_name = {}
            if hasattr(self, 'batter_stats'):
                for _, row in self.batter_stats.iterrows():
                    if 'IDfg' in row and 'Name' in row:
                        batter_id_to_name[int(row['IDfg'])] = row['Name']
                logger.info(f"Mapped {len(batter_id_to_name)} batter IDs to names")

            # Get most common stance for each batter from pitch data
            stance_data = self.merged_pitch_data[
                (self.merged_pitch_data['stand'].notna()) &
                (self.merged_pitch_data['batter_fg_id'].notna())
            ].copy()

            batter_stances = {}

            # Group by batter FG ID and stance, count occurrences
            if len(stance_data) > 0:
                stance_counts = stance_data.groupby(['batter_fg_id', 'stand']).size().reset_index(name='count')

                # For each batter, find all their stances and counts
                for batter_id in stance_counts['batter_fg_id'].unique():
                    batter_stances_df = stance_counts[stance_counts['batter_fg_id'] == batter_id]

                    # Check if batter uses multiple stances (switch hitter)
                    stances_used = batter_stances_df['stand'].unique()

                    # Get the batter name
                    batter_name = batter_id_to_name.get(int(batter_id))
                    if not batter_name:
                        continue

                    # If batter uses both R and L, mark as switch hitter
                    if len(stances_used) > 1 and set(['R', 'L']).issubset(set(stances_used)):
                        batter_stances[batter_name] = 'S'  # S for Switch hitter
                    else:
                        # Otherwise, use most common stance
                        most_common = batter_stances_df.loc[batter_stances_df['count'].idxmax(), 'stand']
                        batter_stances[batter_name] = most_common

                stances_path = os.path.join(self.output_dir, 'batter_stances.json')
                with open(stances_path, 'w') as f:
                    json.dump(batter_stances, f, indent=2)
                logger.info(f"Saved {len(batter_stances)} batter stances to {stances_path}")

                # Log switch hitters
                switch_hitters = [name for name, stance in batter_stances.items() if stance == 'S']
                logger.info(f"Found {len(switch_hitters)} switch hitters")
            else:
                logger.warning("No stance data available to create batter stance mapping")

            # Create pitcher throwing hand mapping
            logger.info("Creating pitcher throwing hand mapping from pitch data...")
            pitcher_throws = {}
            if 'p_throws' in self.merged_pitch_data.columns:
                throws_counts = self.merged_pitch_data.dropna(subset=['player_name', 'p_throws']).groupby(['player_name', 'p_throws']).size().reset_index(name='count')
                for player_name in throws_counts['player_name'].unique():
                    df_p = throws_counts[throws_counts['player_name'] == player_name]
                    most_common = df_p.loc[df_p['count'].idxmax(), 'p_throws']
                    pitcher_throws[player_name] = most_common
                    if ', ' in player_name:
                        parts = player_name.split(', ')
                        pitcher_throws[f"{parts[1]} {parts[0]}"] = most_common

                throws_path = os.path.join(self.output_dir, 'pitcher_throws.json')
                with open(throws_path, 'w') as f:
                    json.dump(pitcher_throws, f, indent=2)
                logger.info(f"Saved {len(pitcher_throws)} pitcher throws to {throws_path}")
        else:
            logger.warning("Could not create batter stance mapping - merged data not available")

        # NEW: Save pitch data CSV for database visualization
        logger.info("Saving pitch data CSV for database...")
        if hasattr(self, 'pitch_data') and len(self.pitch_data) > 0:
            csv_path = os.path.join(self.output_dir, f'pitching_data_{self.start_date}_to_{self.end_date}.csv')
            self.pitch_data.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(self.pitch_data)} pitches to {csv_path}")
        else:
            logger.warning("No pitch data available to save as CSV")
    
    def run(self):
        """Execute full pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Mapping Heat Pipeline")
        logger.info("=" * 60)
        
        # Test network connectivity first
        logger.info("\nStep 0: Testing network connectivity...")
        if not test_network_connection():
            logger.error("\n❌ PIPELINE FAILED: Cannot connect to data sources")
            logger.error("Please fix the SSL/network issue and try again")
            return
        
        # Extract years from date range
        start_year = int(self.start_date.split('-')[0])
        end_year = int(self.end_date.split('-')[0])
        years = list(range(start_year, end_year + 1))
        
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")
        logger.info(f"  Seasons: {years}")
        logger.info(f"  Output: {self.output_dir}")
        
        # Step 1: Fetch season stats for all years
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Fetching season statistics")
        logger.info("=" * 60)
        self.batter_stats, self.pitcher_stats = self.fetch_season_stats(years)
        
        # Step 2: Calculate league averages (from all years combined)
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Calculating league averages")
        logger.info("=" * 60)
        self.league_averages = self.calculate_league_averages(
            self.batter_stats, self.pitcher_stats
        )
        
        # Step 3: Fetch ID mapping
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Fetching player ID mappings")
        logger.info("=" * 60)
        self.id_mapping = self.fetch_id_mapping()
        
        # Step 4: Fetch Statcast data
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Fetching Statcast pitch data")
        logger.info("=" * 60)
        self.pitch_data = self.fetch_statcast_bulk()
        
        # Step 5: Merge context
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Merging season context onto pitches")
        logger.info("=" * 60)
        merged_data = self.merge_context(
            self.pitch_data, self.batter_stats, self.pitcher_stats, self.id_mapping
        )
        
        # Preserve merged data for pitch profiles (before feature engineering modifies it)
        self.merged_pitch_data = merged_data.copy()
        
        # Step 6: Engineer features
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Engineering features")
        logger.info("=" * 60)
        model_data = self.engineer_features(merged_data)
        
        # Step 7: Train model
        logger.info("\n" + "=" * 60)
        logger.info("Step 7: Training LightGBM model")
        logger.info("=" * 60)
        self.model = self.train_model(model_data)
        
        # Step 8: Save artifacts (use most recent year for rosters)
        logger.info("\n" + "=" * 60)
        logger.info("Step 8: Saving artifacts")
        logger.info("=" * 60)
        self.save_artifacts()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\nArtifacts saved to: {os.path.abspath(self.output_dir)}")
        logger.info("\nNext steps:")
        logger.info("  1. Run 'docker-compose up --build' to start services")
        logger.info("  2. Open http://localhost in your browser")
        logger.info("  3. Enjoy Mapping Heat! ⚾")


if __name__ == "__main__":
    # Train on 2025 season (May through October) for fast development & testing
    # This will fetch season stats and all pitches from the date range with sequence features
    pipeline = MappingHeatPipeline(
        start_date="2025-05-01",  # Mid-2025 season
        end_date="2025-10-31",     # End of 2025 season (through playoffs)
        output_dir="artifacts"
    )
    
    pipeline.run()