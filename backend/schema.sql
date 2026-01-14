DROP TABLE IF EXISTS pitching_data;

CREATE TABLE pitching_data (
    pitch_type              TEXT,
    game_date               TEXT,
    release_speed           REAL,
    release_pos_x           REAL,
    release_pos_z           REAL,
    player_name             TEXT,
    batter                  REAL, 
    pitcher                 REAL, 
    events                  TEXT,
    description             TEXT,
    zone                    REAL, 
    stand                   TEXT,
    p_throws                TEXT,
    type                    TEXT,
    balls                   REAL,
    strikes                 REAL,
    pfx_x                   REAL,
    pfx_z                   REAL,
    plate_x                 REAL,
    plate_z                 REAL,
    on_3b                   REAL, 
    on_2b                   REAL, 
    on_1b                   REAL, 
    outs_when_up            REAL,
    inning                  REAL,
    sz_top                  REAL,
    sz_bot                  REAL,
    effective_speed         REAL,
    release_spin_rate       REAL,
    release_extension       REAL,
    spin_axis               REAL, 
    arm_angle               REAL, 
    n_thruorder_pitcher     REAL
);

CREATE INDEX idx_player_name ON pitching_data(player_name);
CREATE INDEX idx_pitch_type ON pitching_data(pitch_type);