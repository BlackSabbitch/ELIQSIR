import os
import mysql.connector
from mysql.connector import Error, pooling
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables from the .env file
load_dotenv()

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

# Initialize the connection manager instance for import
db_manager = ConnectionManager()