"""Data warehouse loader.

Loads cleaned dimensional CSV files into the MySQL star schema in AWS RDS.
Uses the custom ConnectionManager and mysql.connector's executemany() 
for high-performance, memory-efficient upserts without staging tables.
"""

import pandas as pd
from pathlib import Path
import numpy as np

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