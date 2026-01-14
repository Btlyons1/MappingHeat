"""
Mapping Heat Inference Engine
Loads artifacts and provides prediction interface.
"""
import json
import pickle
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MappingHeatModel:
    """Production inference engine for Mapping Heat predictions."""
    
    def __init__(self, artifacts_dir: str = 'artifacts'):
        """
        Initialize model with artifacts.
        
        Args:
            artifacts_dir: Directory containing saved artifacts
        """
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.label_encoders = {}
        self.batters = {}
        self.pitchers = {}
        self.league_averages = {}
        self.pitch_profiles = {}
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all artifacts from disk."""
        logger.info("Loading artifacts")
        
        # Load model and encoders
        try:
            with open(f'{self.artifacts_dir}/lgbm_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_encoders = data['label_encoders']
            logger.info("Loaded model successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load batters
        try:
            with open(f'{self.artifacts_dir}/batters.json', 'r') as f:
                batters_list = json.load(f)
                self.batters = {b['Name']: b for b in batters_list}
            logger.info(f"Loaded {len(self.batters)} batters")
        except Exception as e:
            logger.error(f"Failed to load batters: {e}")
            raise
        
        # Load pitchers
        try:
            with open(f'{self.artifacts_dir}/pitchers.json', 'r') as f:
                pitchers_list = json.load(f)
                self.pitchers = {p['Name']: p for p in pitchers_list}
            logger.info(f"Loaded {len(self.pitchers)} pitchers")
        except Exception as e:
            logger.error(f"Failed to load pitchers: {e}")
            raise
        
        # Load league averages
        try:
            with open(f'{self.artifacts_dir}/league_averages.json', 'r') as f:
                self.league_averages = json.load(f)
            logger.info(f"Loaded {len(self.league_averages)} league averages")
        except Exception as e:
            logger.error(f"Failed to load league averages: {e}")
            raise
        
        # Load pitch profiles
        try:
            with open(f'{self.artifacts_dir}/pitcher_pitch_profiles.json', 'r') as f:
                self.pitch_profiles = json.load(f)
            logger.info(f"Loaded pitch profiles for {len(self.pitch_profiles)} pitchers")
        except Exception as e:
            logger.warning(f"Failed to load pitch profiles: {e}")
            self.pitch_profiles = {}
    
    def get_pitch_profile(self, pitcher_name: Optional[str], pitch_type: str) -> Dict:
        """
        Get average pitch characteristics for a specific pitcher and pitch type.
        
        Args:
            pitcher_name: Name of pitcher or None for league average
            pitch_type: Pitch type code (FF, SL, etc.)
            
        Returns:
            Dictionary of pitch characteristics
        """
        # League average defaults
        defaults = {
            'FF': {'release_speed': 93.0, 'pfx_x': 0.0, 'pfx_z': 10.0, 'release_spin_rate': 2200, 
                   'release_extension': 6.0, 'spin_axis': 180, 'arm_angle': 45},
            'SI': {'release_speed': 92.5, 'pfx_x': 8.0, 'pfx_z': 5.0, 'release_spin_rate': 2100,
                   'release_extension': 6.0, 'spin_axis': 225, 'arm_angle': 45},
            'SL': {'release_speed': 84.0, 'pfx_x': 3.0, 'pfx_z': 1.0, 'release_spin_rate': 2400,
                   'release_extension': 6.0, 'spin_axis': 90, 'arm_angle': 45},
            'CH': {'release_speed': 84.5, 'pfx_x': -6.0, 'pfx_z': 6.0, 'release_spin_rate': 1700,
                   'release_extension': 6.0, 'spin_axis': 270, 'arm_angle': 45},
            'CU': {'release_speed': 78.0, 'pfx_x': -5.0, 'pfx_z': -5.0, 'release_spin_rate': 2500,
                   'release_extension': 6.0, 'spin_axis': 135, 'arm_angle': 45},
            'FC': {'release_speed': 88.0, 'pfx_x': -2.0, 'pfx_z': 8.0, 'release_spin_rate': 2300,
                   'release_extension': 6.0, 'spin_axis': 200, 'arm_angle': 45},
        }
        
        # Try to get pitcher-specific profile
        if pitcher_name and pitcher_name in self.pitch_profiles:
            pitcher_profiles = self.pitch_profiles[pitcher_name]
            if pitch_type in pitcher_profiles:
                return pitcher_profiles[pitch_type]
        
        # Fall back to league average for this pitch type
        return defaults.get(pitch_type, defaults['FF'])
    
    def get_batter_stats(self, batter_name: Optional[str]) -> Dict:
        """
        Get batter stats, falling back to league average if not found.
        
        Args:
            batter_name: Name of batter or None for league average
            
        Returns:
            Dictionary of batter stats
        """
        if batter_name and batter_name in self.batters:
            return self.batters[batter_name]
        
        # Return league averages
        return {
            'wOBA': self.league_averages.get('batter_wOBA', 0.320),
            'OBP': self.league_averages.get('batter_OBP', 0.320),
            'SLG': self.league_averages.get('batter_SLG', 0.415),
            'BB%': self.league_averages.get('batter_BB%', 8.5),
            'K%': self.league_averages.get('batter_K%', 22.0)
        }
    
    def get_pitcher_stats(self, pitcher_name: Optional[str]) -> Dict:
        """
        Get pitcher stats, falling back to league average if not found.
        
        Args:
            pitcher_name: Name of pitcher or None for league average
            
        Returns:
            Dictionary of pitcher stats
        """
        if pitcher_name and pitcher_name in self.pitchers:
            return self.pitchers[pitcher_name]
        
        # Return league averages
        return {
            'FIP': self.league_averages.get('pitcher_FIP', 4.00),
            'ERA': self.league_averages.get('pitcher_ERA', 4.20),
            'WHIP': self.league_averages.get('pitcher_WHIP', 1.30),
            'K/9': self.league_averages.get('pitcher_K/9', 8.5),
            'BB/9': self.league_averages.get('pitcher_BB/9', 3.0)
        }
    
    def predict(self, pitch_data: Dict, batter_name: Optional[str] = None,
                pitcher_name: Optional[str] = None) -> float:
        """
        Predict probability of hit given pitch characteristics and matchup.
        
        Args:
            pitch_data: Dictionary with pitch physics (release_speed, pfx_x, etc.)
            batter_name: Name of batter (None for league average)
            pitcher_name: Name of pitcher (None for league average)
            
        Returns:
            Probability of hit (0-1)
        """
        # Get player stats
        batter_stats = self.get_batter_stats(batter_name)
        pitcher_stats = self.get_pitcher_stats(pitcher_name)
        
        # Build feature vector
        features = {
            # Pitch physics
            'release_speed': pitch_data.get('release_speed', 92.0),
            'pfx_x': pitch_data.get('pfx_x', 0.0),
            'pfx_z': pitch_data.get('pfx_z', 0.0),
            'release_spin_rate': pitch_data.get('release_spin_rate', 2200),
            'release_extension': pitch_data.get('release_extension', 6.0),
            'plate_x': pitch_data.get('plate_x', 0.0),
            'plate_z': pitch_data.get('plate_z', 2.5),
            'effective_speed': pitch_data.get('effective_speed', 92.0),
            'spin_axis': pitch_data.get('spin_axis', 180),
            'release_pos_x': pitch_data.get('release_pos_x', 0.0),
            'release_pos_z': pitch_data.get('release_pos_z', 6.0),
            'arm_angle': pitch_data.get('arm_angle', 45),
            
            # Count
            'balls': pitch_data.get('balls', 0),
            'strikes': pitch_data.get('strikes', 0),
            
            # Zone
            'sz_top': pitch_data.get('sz_top', 3.5),
            'sz_bot': pitch_data.get('sz_bot', 1.5),
            
            # Baserunners
            'on_1b': pitch_data.get('on_1b', 0),
            'on_2b': pitch_data.get('on_2b', 0),
            'on_3b': pitch_data.get('on_3b', 0),
            
            # Game state
            'outs_when_up': pitch_data.get('outs_when_up', 0),
            'inning': pitch_data.get('inning', 1),
            'n_thruorder_pitcher': pitch_data.get('n_thruorder_pitcher', 1),
            
            # Categorical (encoded)
            'pitch_type': self._encode_categorical('pitch_type', pitch_data.get('pitch_type', 'FF')),
            'zone': pitch_data.get('zone', 5),
            'stand': self._encode_categorical('stand', pitch_data.get('stand', 'R')),
            'p_throws': self._encode_categorical('p_throws', pitch_data.get('p_throws', 'R')),
        }
        
        # Add batter stats
        features['batter_wOBA'] = batter_stats.get('wOBA', 0.320)
        features['batter_OBP'] = batter_stats.get('OBP', 0.320)
        features['batter_SLG'] = batter_stats.get('SLG', 0.415)
        features['batter_BB%'] = batter_stats.get('BB%', 8.5)
        features['batter_K%'] = batter_stats.get('K%', 22.0)
        
        # Add pitcher stats
        features['pitcher_FIP'] = pitcher_stats.get('FIP', 4.00)
        features['pitcher_ERA'] = pitcher_stats.get('ERA', 4.20)
        features['pitcher_WHIP'] = pitcher_stats.get('WHIP', 1.30)
        features['pitcher_K/9'] = pitcher_stats.get('K/9', 8.5)
        features['pitcher_BB/9'] = pitcher_stats.get('BB/9', 3.0)
        
        # Convert to DataFrame for prediction
        X = pd.DataFrame([features])
        
        # Align with training features
        X = X.reindex(columns=self.model.feature_name_, fill_value=0)
        
        # Predict probability
        prob = self.model.predict_proba(X)[0][1]
        
        return float(prob)
    
    def _encode_categorical(self, col_name: str, value: str) -> int:
        """
        Encode categorical value using saved label encoder.
        
        Args:
            col_name: Column name
            value: Value to encode
            
        Returns:
            Encoded integer
        """
        if col_name in self.label_encoders:
            le = self.label_encoders[col_name]
            try:
                return int(le.transform([value])[0])
            except ValueError:
                # Unknown category, return 0
                return 0
        return 0
    
    def get_rosters(self) -> Dict:
        """
        Get lists of available batters and pitchers with stance information.

        Returns:
            Dictionary with 'batters', 'pitchers', and 'batter_stances'
        """
        # Create batter stance mapping from historical data
        # This requires loading the stance mapping artifact if available
        stances = {}

        try:
            import json
            import os
            stance_file = os.path.join(self.artifacts_dir, 'batter_stances.json')
            if os.path.exists(stance_file):
                with open(stance_file, 'r') as f:
                    stances = json.load(f)
                logger.info(f"Loaded {len(stances)} batter stances from file")
            else:
                logger.warning(f"Batter stance file not found: {stance_file}")
        except Exception as e:
            logger.warning(f"Could not load batter stances: {e}")

        return {
            'batters': sorted(list(self.batters.keys())),
            'pitchers': sorted(list(self.pitchers.keys())),
            'batter_stances': stances
        }