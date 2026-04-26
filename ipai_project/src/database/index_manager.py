import logging
from src.database.connection_manager import db_manager

class IndexManager:
    """
    Manages database indexes for the ELIQSIR DWH to optimize ETL and IR performance.
    Handles creation, targeted drops, and maintenance of search-related indexes.
    """

    def __init__(self):
        # Dictionary of critical indexes for our specific Star Schema
        self.required_indexes = {
            "idx_fact_article": ("fact_bioactivity", "article_key"),
            "idx_fact_drug": ("fact_bioactivity", "drug_key"),
            "idx_fact_protein": ("fact_bioactivity", "protein_key"),
            "idx_fact_pub_date": ("fact_bioactivity", "publication_date_key"),
            "idx_fact_acc_date": ("fact_bioactivity", "accepted_date_key"),
            "idx_fact_rec_date": ("fact_bioactivity", "received_date_key"),
            "idx_article_pubmed": ("dim_article", "pubmed_id"),
            "idx_struct_protein": ("dim_structure", "protein_key")
        }
        self.logger = logging.getLogger(__name__)

    def create_required_indexes(self):
        """Creates all indexes necessary for high-performance extraction."""
        print("Creating optimization indexes...")
        for idx_name, (table, column) in self.required_indexes.items():
            query = f"CREATE INDEX {idx_name} ON {table}({column})"
            try:
                with db_manager.get_dwh_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(query)
                print(f"Created: {idx_name} on {table}({column})")
            except Exception as e:
                if "Duplicate key name" in str(e) or "already exists" in str(e).lower():
                    print(f"Skipped: {idx_name} (already exists)")
                else:
                    print(f"Error creating {idx_name}: {e}")

    def drop_index(self, index_name, table_name):
        """Drops a specific index from a table."""
        query = f"DROP INDEX {index_name} ON {table_name}"
        try:
            with db_manager.get_dwh_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
            print(f"Dropped: {index_name}")
        except Exception as e:
            print(f"Failed to drop {index_name}: {e}")

    def drop_all_custom_indexes(self):
        """
        Drops all indexes defined in self.required_indexes.
        Useful for resetting the DB state or before bulk data loads.
        """
        print("Cleaning up custom indexes...")
        for idx_name, (table, _) in self.required_indexes.items():
            self.drop_index(idx_name, table)

    def maintenance_drop_non_essential(self):
        """
        ADVANCED: Drops all indexes in the database EXCEPT Primary Keys.
        Use with caution only during massive rebuilds.
        """
        print("Dropping all non-Primary Key indexes...")
        # Query to find all non-primary indexes in MySQL/RDS
        find_idx_query = """
            SELECT DISTINCT INDEX_NAME, TABLE_NAME 
            FROM information_schema.STATISTICS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND INDEX_NAME != 'PRIMARY'
        """
        try:
            with db_manager.get_dwh_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(find_idx_query)
                    indexes = cursor.fetchall()
                    
            for idx_name, table in indexes:
                self.drop_index(idx_name, table)
        except Exception as e:
            print(f"Maintenance drop failed: {e}")

    