import os
import mysql.connector
from mysql.connector import Error, pooling
from dotenv import load_dotenv
from contextlib import contextmanager
import pandas as pd
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)
class ConnectionManager:
    def __init__(self):
        # Target DWH (AWS RDS) configuration parameters
        self.dwh_config = {
            'host': os.getenv('MYSQL_DWH_HOST'),
            'port': int(os.getenv('MYSQL_DWH_PORT', 3306)),
            'user': os.getenv('MYSQL_DWH_USER'),
            'password': os.getenv('MYSQL_DWH_PASSWORD'),
            'database': os.getenv('MYSQL_DWH_NAME')
        }
        
        # Source configuration (Local ChEMBL MySQL, if used)
        self.source_config = {
            'host': os.getenv('MYSQL_HOST'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('CHEMBL_MYSQL_DB')
        }

    @contextmanager
    def get_dwh_connection(self):
        """Context manager for connecting to the AWS DWH."""
        conn = None
        try:
            conn = mysql.connector.connect(**self.dwh_config)
            if conn.is_connected():
                yield conn
        except Error as e:
            print(f"Error connecting to AWS DWH: {e}")
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()

    @contextmanager
    def get_source_connection(self):
        """Context manager for the local ChEMBL database."""
        conn = None
        try:
            conn = mysql.connector.connect(**self.source_config)
            if conn.is_connected():
                yield conn
        except Error as e:
            print(f"Error connecting to the local database: {e}")
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()

    def fetch_to_dataframe(self, query: str) -> pd.DataFrame:
        """Executes a query and returns a pandas DataFrame directly."""
        with self.get_dwh_connection() as conn:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                return pd.read_sql(query, con=conn)
            

# Initialize the connection manager instance for import
db_manager = ConnectionManager()