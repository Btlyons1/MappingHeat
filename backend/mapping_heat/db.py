import sqlite3
import click
import os
from flask import current_app, g, Flask
from flask.cli import with_appcontext
from sqlite3 import Connection
from typing import Optional
import pandas as pd

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

START_DATE = '2023-03-30' 
END_DATE = '2024-9-29'

CSV_FILENAME = f"pitching_data_{START_DATE}_to_{END_DATE}.csv"
print(f"File name is: {CSV_FILENAME}")

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CURRENT_CSV_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "mapping_heat", "fixtures", CSV_FILENAME)

def init_db() -> None:
    """Initialize the database using schema.sql and load data from the current v5 CSV."""
    db: Connection = get_db()
    instance_path: str = current_app.instance_path

    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: Schema file not found at {SCHEMA_PATH}. Cannot initialize DB schema.")
    else:
        print(f"Initializing database at {current_app.config['DATABASE']} using schema {SCHEMA_PATH}")
        try:
            with current_app.open_resource(SCHEMA_PATH) as f:
                db.executescript(f.read().decode('utf8'))
            print("Database schema applied.")
        except Exception as e:
            print(f"An unexpected error occurred during DB schema initialization: {e}")

    print(f"Attempting to populate DB using current CSV: {CURRENT_CSV_DATA_PATH}")
    if os.path.exists(CURRENT_CSV_DATA_PATH):
        db_file_path: str = current_app.config['DATABASE']
        try:
            os.makedirs(instance_path, exist_ok=True)
        except OSError:
            pass
        try:
            df_from_csv = pd.read_csv(CURRENT_CSV_DATA_PATH, low_memory=False)

            print(f"Running csv_to_sqlite on {CURRENT_CSV_DATA_PATH} into {db_file_path}...")
            if not df_from_csv.empty:
                print(f"Loaded {len(df_from_csv)} rows from CSV. Writing to 'pitching_data' table...")
                df_from_csv.to_sql("pitching_data", db, if_exists="replace", index=False)
                print("✅ CSV data successfully loaded into 'pitching_data' table.")
            else:
                print("⚠️ CSV file is empty. 'pitching_data' table will be empty.")
        except pd.errors.EmptyDataError:
             print(f"❌ Error: The CSV file at {CURRENT_CSV_DATA_PATH} is empty.")
        except FileNotFoundError:
             print(f"❌ Error: The CSV file at {CURRENT_CSV_DATA_PATH} was not found during pandas read.")
        except Exception as e:
            print(f"Error during pandas CSV read or to_sql operation: {e}")
            print("Please ensure the CSV file is correctly formatted and accessible.")
    else:
        print(f"Data CSV ({CURRENT_CSV_DATA_PATH}) not found. 'pitching_data' table will be empty.")

def get_db() -> Connection:
    if 'db' not in g:
        db_path: str = current_app.config['DATABASE']
        print(f"Connecting to database: {db_path}")
        try:
            g.db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            g.db.row_factory = sqlite3.Row
            print("Database connection established.")
        except sqlite3.Error as e:
            print(f"ERROR: Failed to connect to database at {db_path}: {e}")
            raise
    return g.db

def close_db(e: Optional[Exception] = None) -> None:
    db_conn: Optional[Connection] = g.pop('db', None)
    if db_conn is not None:
        db_conn.close()
        print("Database connection closed.")

@click.command('init-db')
@with_appcontext
def init_db_command() -> None:
    """Clear existing data, create new tables from schema, then populates from current CSV."""
    init_db()
    click.echo('Initialized and attempted to populate the database.')

def init_app(app: Flask) -> None:
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)