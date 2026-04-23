"""Data warehouse loader.

Loads cleaned dimensional CSV files into the MySQL star schema in AWS RDS.
Uses the custom ConnectionManager and mysql.connector's executemany() 
for high-performance, memory-efficient upserts without staging tables.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime  # <--- ДОБАВИТЬ ЭТУ СТРОКУ

from src.database.connection_manager import db_manager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
class WarehouseLoader:
    """Load star-schema CSVs into the MySQL data warehouse.

    Parameters
    ----------
    chunksize:
        Number of rows processed per database transaction. 
        10,000 - 50,000 is a good range for direct executemany inserts.
    """

    def __init__(self, chunksize: int = 10000) -> None:
        self.chunksize = chunksize

    def _upsert_chunk(self, df_chunk: pd.DataFrame, table_name: str, pk_column: str, conn) -> None:
        """Execute a bulk upsert for a single DataFrame chunk."""
        if df_chunk.empty:
            return

        columns = list(df_chunk.columns)

        # 1. Replace pandas NA/NaN with Python None so MySQL receives true NULLs
        df_chunk = df_chunk.astype(object).where(pd.notnull(df_chunk), None)

        # 2. Build the parameterised SQL query
        placeholders = ", ".join(["%s"] * len(columns))
        cols_str = ", ".join(f"`{c}`" for c in columns)
        
        # ON DUPLICATE KEY UPDATE clause
        update_cols = [c for c in columns if c != pk_column]
        updates = ", ".join(f"`{c}` = VALUES(`{c}`)" for c in update_cols)

        sql = f"""
            INSERT INTO `{table_name}` ({cols_str})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {updates};
        """

        # 3. Convert DataFrame to a list of tuples for executemany
        data = [tuple(row) for row in df_chunk.to_numpy()]

        # 4. Execute and commit
        with conn.cursor() as cursor:
            cursor.executemany(sql, data)
        conn.commit()

    def load_table(self, csv_path: Path, table_name: str, pk_column: str) -> int:
        """Read a CSV in chunks and upsert it into the database."""
        if not csv_path.exists():
            logger.error("File not found: %s", csv_path)
            return 0

        logger.info("Starting load for '%s' from %s", table_name, csv_path)
        total_rows = 0

        # Open a single DWH connection for the entire file
        with db_manager.get_dwh_connection() as conn:
            # Read CSV in chunks to save RAM
            with pd.read_csv(csv_path, chunksize=self.chunksize) as reader:
                for i, chunk in enumerate(reader):
                    self._upsert_chunk(chunk, table_name, pk_column, conn)
                    total_rows += len(chunk)
                    
                    if (i + 1) % 5 == 0:
                        logger.info("... loaded %d rows into '%s'", total_rows, table_name)

        logger.info("Finished loading '%s'. Total rows: %d", table_name, total_rows)
        return total_rows

    def load_all(self, csv_dir: str | Path) -> dict[str, int]:
        """Load all dimensions and the fact table in the correct FK order."""
        csv_dir = Path(csv_dir)
        results = {}

        logger.info("Beginning bulk warehouse load from directory: %s", csv_dir)

        # 1. Dimensions MUST be loaded first to establish valid Foreign Keys
        results["dim_protein"] = self.load_table(
            csv_dir / "dim_protein.csv", "dim_protein", "protein_key"
        )
        results["dim_drug"] = self.load_table(
            csv_dir / "dim_drug.csv", "dim_drug", "drug_key"
        )
        results["dim_article"] = self.load_table(
            csv_dir / "dim_article.csv", "dim_article", "article_key"
        )
        
        # Loading the Data-Driven Date Dimension
        results["dim_date"] = self.load_table(
            csv_dir / "dim_date.csv", "dim_date", "date_key"
        )
        
        results["dim_structure"] = self.load_table(
            csv_dir / "dim_structure.csv", "dim_structure", "structure_key"
        )

        # 2. The Fact table is loaded last
        results["fact_bioactivity"] = self.load_table(
            csv_dir / "fact_bioactivity.csv", "fact_bioactivity", "activity_id"
        )

        total = sum(results.values())
        logger.info("Bulk load complete! %d total rows written.", total)
        return results
   
    def update_date_dimension_from_facts(self):
        """
        Scans all lifecycle date columns in fact_bioactivity, 
        identifies missing keys in dim_date, and populates them
        with human-readable attributes.
        """
        print("Scanning fact table for missing date keys...")
        
        union_query = """
        SELECT DISTINCT date_key FROM (
            SELECT received_date_key as date_key FROM fact_bioactivity
            UNION SELECT revised_date_key FROM fact_bioactivity
            UNION SELECT accepted_date_key FROM fact_bioactivity
            UNION SELECT epub_date_key FROM fact_bioactivity
            UNION SELECT ppub_date_key FROM fact_bioactivity
        ) AS all_keys
        WHERE date_key != 19000101 
          AND date_key NOT IN (SELECT date_key FROM dim_date);
        """
        
        with db_manager.get_dwh_connection() as conn:
            missing_keys_df = pd.read_sql(union_query, conn)
            
        missing_keys = missing_keys_df['date_key'].tolist()

        if not missing_keys:
            print("All date keys are already present in dim_date. No update needed.")
            return

        print(f"Found {len(missing_keys)} missing dates. Generating human-readable metadata...")

        new_date_records = []
        for key in missing_keys:
            try:
                date_str = str(int(key))
                dt = datetime.strptime(date_str, '%Y%m%d')
                
                # Calculate fractional year
                fractional_year = round(dt.year + ((dt.timetuple().tm_yday - 1) / 365.25), 3)
                
                new_date_records.append((
                    int(key),                               # date_key
                    dt.date(),                              # full_date
                    dt.strftime('%B %d, %Y, %A'),           # full_date_desc
                    dt.year,                                # year
                    dt.strftime('%B'),                      # month_name
                    dt.day,                                 # day
                    (dt.month - 1) // 3 + 1,                # quarter
                    dt.strftime('%A'),                      # day_name
                    'weekend' if dt.weekday() >= 5 else 'non-weekend', # is_weekend
                    fractional_year,                        # fractional_year
                    int(dt.timestamp())                     # epoch_time
                ))
            except ValueError:
                print(f"Skipping invalid date key found in data: {key}")
                continue

        # Match the exact columns from the updated schema
        insert_sql = """
        INSERT INTO dim_date 
        (date_key, full_date, full_date_desc, year, month_name, day, quarter, day_name, is_weekend, fractional_year, epoch_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with db_manager.get_dwh_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_sql, new_date_records)
            conn.commit()
            
        print(f"Successfully added {len(new_date_records)} human-readable dates to dim_date.")