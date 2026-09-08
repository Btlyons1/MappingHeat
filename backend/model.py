"""
Mapping Heat Inference Engine
Loads artifacts and provides prediction interface.
"""
import os
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
                from collections import Counter
                b_counts = Counter([b['Name'] for b in batters_list])
                dup_names = {k for k, v in b_counts.items() if v > 1}

                self.batters = {}
                for b in batters_list:
                    name = b['Name']
                    if name in dup_names:
                        id_str = f"#{b['MLBAM_ID']}" if b.get('MLBAM_ID') else "Prospect"
                        key_name = f"{name} ({id_str})"
                    else:
                        key_name = name
                    b['raw_name'] = name
                    self.batters[key_name] = b
                    # Also map raw name for fallback lookups (preferring valid MLBAM_ID)
                    if name not in self.batters or (self.batters[name].get('MLBAM_ID') is None and b.get('MLBAM_ID') is not None):
                        self.batters[name] = b
            logger.info(f"Loaded {len(self.batters)} batter keys")
        except Exception as e:
            logger.error(f"Failed to load batters: {e}")
            raise
        
        # Load pitchers
        try:
            with open(f'{self.artifacts_dir}/pitchers.json', 'r') as f:
                pitchers_list = json.load(f)
                from collections import Counter
                p_counts = Counter([p['Name'] for p in pitchers_list])
                dup_p_names = {k for k, v in p_counts.items() if v > 1}

                self.pitchers = {}
                for p in pitchers_list:
                    name = p['Name']
                    if name in dup_p_names:
                        id_str = f"#{p['MLBAM_ID']}" if p.get('MLBAM_ID') else "Prospect"
                        key_name = f"{name} ({id_str})"
                    else:
                        key_name = name
                    p['raw_name'] = name
                    self.pitchers[key_name] = p
                    if name not in self.pitchers or (self.pitchers[name].get('MLBAM_ID') is None and p.get('MLBAM_ID') is not None):
                        self.pitchers[name] = p
            logger.info(f"Loaded {len(self.pitchers)} pitcher keys")
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
                raw_profiles = json.load(f)
            # Map "Last, First" keys to "First Last" to match incoming query parameters
            self.pitch_profiles = {}
            for name_last_first, profile in raw_profiles.items():
                if ',' in name_last_first:
                    parts = name_last_first.split(',')
                    last_name = parts[0].strip()
                    first_name = parts[1].strip()
                    name_first_last = f"{first_name} {last_name}"
                    self.pitch_profiles[name_first_last] = profile
                else:
                    self.pitch_profiles[name_last_first] = profile
            logger.info(f"Loaded and mapped pitch profiles for {len(self.pitch_profiles)} pitchers")
        except Exception as e:
            logger.warning(f"Failed to load pitch profiles: {e}")
            self.pitch_profiles = {}

        # Load pitcher repertoire percentages
        try:
            repertoire_path = f'{self.artifacts_dir}/pitcher_repertoire.json'
            if os.path.exists(repertoire_path):
                with open(repertoire_path, 'r') as f:
                    self.repertoire = json.load(f)
                logger.info(f"Loaded pitcher repertoire for {len(self.repertoire)} pitchers")
            else:
                logger.warning(f"Pitcher repertoire file not found: {repertoire_path}")
                self.repertoire = {}
        except Exception as e:
            logger.warning(f"Failed to load pitcher repertoire: {e}")
            self.repertoire = {}
    
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
            'BB%': self.league_averages.get('batter_BB%', 0.085),
            'K%': self.league_averages.get('batter_K%', 0.220),
            'ISO': self.league_averages.get('batter_ISO', 0.160),
            'BABIP': self.league_averages.get('batter_BABIP', 0.295),
            'HardHit%': self.league_averages.get('batter_HardHit%', 0.400),
            'Barrel%': self.league_averages.get('batter_Barrel%', 0.080),
            'Contact%': self.league_averages.get('batter_Contact%', 0.760),
            'O-Swing%': self.league_averages.get('batter_O-Swing%', 0.320)
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
            'BB/9': self.league_averages.get('pitcher_BB/9', 3.0),
            'HR/9': self.league_averages.get('pitcher_HR/9', 1.2),
            'BABIP': self.league_averages.get('pitcher_BABIP', 0.290),
            'xFIP': self.league_averages.get('pitcher_xFIP', 4.10),
            'xERA': self.league_averages.get('pitcher_xERA', 4.25),
            'SwStr%': self.league_averages.get('pitcher_SwStr%', 0.110),
            'FB_velocity': 93.0
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
        
        # Extract variables for calculations
        stand = pitch_data.get('stand', 'R')
        p_throws = pitch_data.get('p_throws', 'R')
        balls = pitch_data.get('balls', 0)
        strikes = pitch_data.get('strikes', 0)
        release_speed = pitch_data.get('release_speed', 92.0)
        effective_speed = pitch_data.get('effective_speed', 92.0)
        plate_x = pitch_data.get('plate_x', 0.0)
        plate_z = pitch_data.get('plate_z', 2.5)

        # Sequence & Tunneling context calculations
        prev_pitch_type = pitch_data.get('prev_pitch_type', 'None')
        pitch_number = pitch_data.get('pitch_number', 1)
        prev_release_speed = pitch_data.get('prev_release_speed')
        prev_plate_x = pitch_data.get('prev_plate_x')
        prev_plate_z = pitch_data.get('prev_plate_z')

        if prev_release_speed is not None and prev_release_speed != '':
            prev_pitch_speed_diff = release_speed - float(prev_release_speed)
        else:
            prev_pitch_speed_diff = 0.0

        if prev_plate_x is not None and prev_plate_x != '' and prev_plate_z is not None and prev_plate_z != '':
            prev_pitch_location_dist = np.sqrt(
                (plate_x - float(prev_plate_x))**2 + 
                (plate_z - float(prev_plate_z))**2
            )
        else:
            prev_pitch_location_dist = 0.0

        # Build feature vector
        features = {
            # Pitch physics
            'release_speed': release_speed,
            'pfx_x': pitch_data.get('pfx_x', 0.0),
            'pfx_z': pitch_data.get('pfx_z', 0.0),
            'release_spin_rate': pitch_data.get('release_spin_rate', 2200),
            'release_extension': pitch_data.get('release_extension', 6.0),
            'plate_x': plate_x,
            'plate_z': plate_z,
            'effective_speed': effective_speed,
            'spin_axis': pitch_data.get('spin_axis', 180),
            'release_pos_x': pitch_data.get('release_pos_x', 0.0),
            'release_pos_z': pitch_data.get('release_pos_z', 6.0),
            'arm_angle': pitch_data.get('arm_angle', 45),
            
            # Count
            'balls': balls,
            'strikes': strikes,
            
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
            'stand': self._encode_categorical('stand', stand),
            'p_throws': self._encode_categorical('p_throws', p_throws),
            
            # Count context features (replicated from pipeline)
            'is_hitters_count': 1 if (balls >= 2 and strikes <= 1) else 0,
            'is_pitchers_count': 1 if (strikes == 2 and balls <= 1) else 0,
            'count_leverage': (balls - strikes) / 3.0,
            'two_strike_count': 1 if strikes == 2 else 0,
            
            # Advanced baseball features
            'is_platoon_advantage': 1 if stand != p_throws else 0,
            'perceived_speed_diff': effective_speed - release_speed,
            'speed_x_location': release_speed * abs(plate_x),
            'speed_x_height': release_speed * plate_z,
            'velocity_differential': pitcher_stats.get('FB_velocity', 93.0) - release_speed,
            'pitcher_avg_fb_speed': pitcher_stats.get('FB_velocity', 93.0),
            
            # Sequence & Tunneling context features
            'pitch_number': pitch_number,
            'prev_pitch_type': self._encode_categorical('prev_pitch_type', prev_pitch_type),
            'prev_pitch_speed_diff': prev_pitch_speed_diff,
            'prev_pitch_location_dist': prev_pitch_location_dist
        }
        
        # Add batter stats
        features['batter_wOBA'] = batter_stats.get('wOBA', 0.320)
        features['batter_OBP'] = batter_stats.get('OBP', 0.320)
        features['batter_SLG'] = batter_stats.get('SLG', 0.415)
        features['batter_BB%'] = batter_stats.get('BB%', 0.085)
        features['batter_K%'] = batter_stats.get('K%', 0.220)
        features['batter_ISO'] = batter_stats.get('ISO', 0.160)
        features['batter_BABIP'] = batter_stats.get('BABIP', 0.295)
        features['batter_HardHit%'] = batter_stats.get('HardHit%', 0.400)
        features['batter_Barrel%'] = batter_stats.get('Barrel%', 0.080)
        features['batter_Contact%'] = batter_stats.get('Contact%', 0.760)
        features['batter_O-Swing%'] = batter_stats.get('O-Swing%', 0.320)
        
        # Add pitcher stats
        features['pitcher_FIP'] = pitcher_stats.get('FIP', 4.00)
        features['pitcher_ERA'] = pitcher_stats.get('ERA', 4.20)
        features['pitcher_WHIP'] = pitcher_stats.get('WHIP', 1.30)
        features['pitcher_K/9'] = pitcher_stats.get('K/9', 8.5)
        features['pitcher_BB/9'] = pitcher_stats.get('BB/9', 3.0)
        features['pitcher_HR/9'] = pitcher_stats.get('HR/9', 1.2)
        features['pitcher_BABIP'] = pitcher_stats.get('BABIP', 0.290)
        features['pitcher_xFIP'] = pitcher_stats.get('xFIP', 4.10)
        features['pitcher_xERA'] = pitcher_stats.get('xERA', 4.25)
        features['pitcher_SwStr%'] = pitcher_stats.get('SwStr%', 0.110)
        
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
        Get lists of available batters and pitchers with stance information and MLBAM IDs.

        Returns:
            Dictionary with 'batters', 'pitchers', 'batter_stances', 'batter_ids', 'pitcher_ids'
        """
        # Create batter stance mapping from historical data
        stances = {}

        try:
            import json
            import os
            stance_file = os.path.join(self.artifacts_dir, 'batter_stances.json')
            if os.path.exists(stance_file):
                with open(stance_file, 'r') as f:
                    raw_stances = json.load(f)
                
                # Map stances for both key names and raw names
                for k, b_obj in self.batters.items():
                    raw_name = b_obj.get('raw_name', k)
                    if raw_name in raw_stances:
                        stances[k] = raw_stances[raw_name]
                        stances[raw_name] = raw_stances[raw_name]
                logger.info(f"Loaded {len(stances)} batter stance mappings")
            else:
                logger.warning(f"Batter stance file not found: {stance_file}")
        except Exception as e:
            logger.warning(f"Could not load batter stances: {e}")

        # Create pitcher throws mapping from historical data
        pitcher_throws = {}
        try:
            import json
            import os
            throws_file = os.path.join(self.artifacts_dir, 'pitcher_throws.json')
            if os.path.exists(throws_file):
                with open(throws_file, 'r') as f:
                    raw_throws = json.load(f)
                
                # Map throws for both key names and raw names
                for k, p_obj in self.pitchers.items():
                    raw_name = p_obj.get('raw_name', k)
                    if raw_name in raw_throws:
                        pitcher_throws[k] = raw_throws[raw_name]
                        pitcher_throws[raw_name] = raw_throws[raw_name]
                    elif k in raw_throws:
                        pitcher_throws[k] = raw_throws[k]
                logger.info(f"Loaded {len(pitcher_throws)} pitcher throws mappings")
            else:
                logger.warning(f"Pitcher throws file not found: {throws_file}")
        except Exception as e:
            logger.warning(f"Could not load pitcher throws: {e}")

        # Build name to MLBAM ID mappings
        batter_ids = {name: b.get('MLBAM_ID') for name, b in self.batters.items() if b.get('MLBAM_ID') is not None}
        pitcher_ids = {name: p.get('MLBAM_ID') for name, p in self.pitchers.items() if p.get('MLBAM_ID') is not None}

        def get_display_name(key_name, player_obj):
            name = player_obj.get('Name', key_name)
            suffix_info = ""
            if "(" in key_name and ")" in key_name:
                suffix_info = " " + key_name[key_name.find("("):]

            parts = name.strip().split()
            if len(parts) <= 1:
                return name + suffix_info
            suffixes = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v'}
            if parts[-1].lower() in suffixes and len(parts) > 2:
                last_name = f"{parts[-2]} {parts[-1]}"
                first_name = " ".join(parts[:-2])
            else:
                last_name = parts[-1]
                first_name = " ".join(parts[:-1])
            return f"{last_name}, {first_name}{suffix_info}"

        # Collect unique player entries (filtering out raw_name aliases if a disambiguated key exists)
        unique_batters_keys = []
        for k in self.batters.keys():
            if '(' in k:
                unique_batters_keys.append(k)
            else:
                id_str_1 = f"#{self.batters[k].get('MLBAM_ID')}" if self.batters[k].get('MLBAM_ID') else "Prospect"
                if f"{k} ({id_str_1})" not in self.batters:
                    unique_batters_keys.append(k)

        unique_pitchers_keys = []
        for k in self.pitchers.keys():
            if '(' in k:
                unique_pitchers_keys.append(k)
            else:
                id_str_1 = f"#{self.pitchers[k].get('MLBAM_ID')}" if self.pitchers[k].get('MLBAM_ID') else "Prospect"
                if f"{k} ({id_str_1})" not in self.pitchers:
                    unique_pitchers_keys.append(k)

        sorted_batters = sorted(
            [{'name': k, 'display_name': get_display_name(k, self.batters[k])} for k in unique_batters_keys],
            key=lambda x: x['display_name'].lower()
        )
        sorted_pitchers = sorted(
            [{'name': k, 'display_name': get_display_name(k, self.pitchers[k])} for k in unique_pitchers_keys],
            key=lambda x: x['display_name'].lower()
        )

        return {
            'batters': sorted_batters,
            'pitchers': sorted_pitchers,
            'batter_stances': stances,
            'pitcher_throws': pitcher_throws,
            'batter_ids': batter_ids,
            'pitcher_ids': pitcher_ids
        }