import pandas as pd
import matplotlib.pyplot as plt
import warnings
from IPython.display import display

from src.utils.logging_config import get_logger

warnings.filterwarnings("ignore", category=UserWarning, message=".*pandas only supports SQLAlchemy.*")

logger = get_logger(__name__)

class EnrichmentValidator:
    """Validates data integrity and visualizes DWH enrichment metrics."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def display_coverage_report(self):
        logger.info("Generating and displaying general fact coverage report...")
        
        query = """
        SELECT 
            'received' AS date_type,
            SUM(CASE WHEN received_date_key != 19000101 THEN 1 ELSE 0 END) AS found_count,
            SUM(CASE WHEN received_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 AS coverage_pct
        FROM fact_bioactivity
        UNION ALL
        SELECT 'revised', SUM(CASE WHEN revised_date_key != 19000101 THEN 1 ELSE 0 END), SUM(CASE WHEN revised_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 FROM fact_bioactivity
        UNION ALL
        SELECT 'accepted', SUM(CASE WHEN accepted_date_key != 19000101 THEN 1 ELSE 0 END), SUM(CASE WHEN accepted_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 FROM fact_bioactivity
        UNION ALL
        SELECT 'epub', SUM(CASE WHEN epub_date_key != 19000101 THEN 1 ELSE 0 END), SUM(CASE WHEN epub_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 FROM fact_bioactivity
        UNION ALL
        SELECT 'ppub', SUM(CASE WHEN ppub_date_key != 19000101 THEN 1 ELSE 0 END), SUM(CASE WHEN ppub_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 FROM fact_bioactivity;
        """
        
        with self.db_manager.get_dwh_connection() as conn:
            summary_df = pd.read_sql(query, conn)

        print("\nGeneral Fact Enriched Data Coverage Report:")
        
        styled_summary = summary_df.style.background_gradient(
            cmap='YlGn', 
            subset=['coverage_pct']
        ).format({'coverage_pct': '{:.2f}%'})
        
        display(styled_summary)

    def display_article_coverage_report(self, article_column="article_key"):
        """Article-level coverage using single-scan aggregation."""
        logger.info(f"Executing article-level audit on {article_column}...")

        query = f"""
        WITH unique_articles AS (
            SELECT 
                MAX(received_date_key) as received,
                MAX(revised_date_key) as revised,
                MAX(accepted_date_key) as accepted,
                MAX(epub_date_key) as epub,
                MAX(ppub_date_key) as ppub
            FROM fact_bioactivity
            GROUP BY {article_column}
        )
        SELECT 
            COUNT(*) as total_count,
            SUM(received != 19000101) as received_count,
            SUM(revised != 19000101) as revised_count,
            SUM(accepted != 19000101) as accepted_count,
            SUM(epub != 19000101) as epub_count,
            SUM(ppub != 19000101) as ppub_count
        FROM unique_articles;
        """

        with self.db_manager.get_dwh_connection() as conn:
            raw_df = pd.read_sql(query, conn)

        # 1. Transform 'wide' format into 'long' format using melt
        total = raw_df['total_count'].iloc[0]
        
        # Drop the total_count from columns we want to transpose
        melt_df = raw_df.drop(columns=['total_count']).melt(
            var_name='date_type', 
            value_name='found_count'
        )
        
        # 2. Clean up names and calculate percentages in Pandas
        melt_df['date_type'] = melt_df['date_type'].str.replace('_count', '')
        melt_df['coverage_pct'] = (melt_df['found_count'] / total) * 100

        print(f"\nOptimized Article-Level Report (Total Unique Articles: {total:,})")
        
        styled_summary = melt_df.style.background_gradient(
            cmap='Blues', 
            subset=['coverage_pct']
        ).format({'coverage_pct': '{:.2f}%'})
        
        display(styled_summary)

    def plot_coverage_trend(self):
        logger.info("Plotting metadata coverage trend...")
        
        query = """
        SELECT 
            LEFT(ppub_date_key, 4) AS pub_year,
            COUNT(*) as total_rows,
            SUM(CASE WHEN received_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 as received_pct,
            SUM(CASE WHEN revised_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 as revised_pct,
            SUM(CASE WHEN accepted_date_key != 19000101 THEN 1 ELSE 0 END) / COUNT(*) * 100 as accepted_pct
        FROM fact_bioactivity
        WHERE ppub_date_key != 19000101 
          AND LEFT(ppub_date_key, 4) >= '1990'
          AND LEFT(ppub_date_key, 4) <= '2023'
        GROUP BY LEFT(ppub_date_key, 4)
        ORDER BY pub_year;
        """

        with self.db_manager.get_dwh_connection() as conn:
            df = pd.read_sql(query, conn)

        df['pub_year'] = pd.to_numeric(df['pub_year'])

        plt.figure(figsize=(14, 7))
        plt.plot(df['pub_year'], df['received_pct'], marker='o', label='Received Date', color='#1f77b4', linewidth=2.5)
        plt.plot(df['pub_year'], df['revised_pct'], marker='s', label='Revised Date', color='#ff7f0e', linewidth=2.5)
        plt.plot(df['pub_year'], df['accepted_pct'], marker='^', label='Accepted Date', color='#2ca02c', linewidth=2.5)
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='ePub / pPub (100% Baseline)')

        plt.title('PubMed Metadata Coverage Trend by Publication Year', fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('Publication Year', fontsize=12, labelpad=10)
        plt.ylabel('Data Coverage (%)', fontsize=12, labelpad=10)
        plt.ylim(0, 110)
        plt.xlim(df['pub_year'].min() - 1, df['pub_year'].max() + 1)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(title='Date Type', title_fontsize='12', fontsize='11', loc='lower right', framealpha=0.9)

        sns_spines = plt.gca().spines
        sns_spines['top'].set_visible(False)
        sns_spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def run_architecture_report(self):
        logger.info("Executing comprehensive architecture report...")
        
        stats_query = """
        SELECT 'Received' as date_type, COUNT(DISTINCT received_date_key) as unique_dates FROM fact_bioactivity WHERE received_date_key != 19000101
        UNION ALL
        SELECT 'Revised', COUNT(DISTINCT revised_date_key) FROM fact_bioactivity WHERE revised_date_key != 19000101
        UNION ALL
        SELECT 'Accepted', COUNT(DISTINCT accepted_date_key) FROM fact_bioactivity WHERE accepted_date_key != 19000101
        UNION ALL
        SELECT 'ePub', COUNT(DISTINCT epub_date_key) FROM fact_bioactivity WHERE epub_date_key != 19000101
        UNION ALL
        SELECT 'pPub', COUNT(DISTINCT ppub_date_key) FROM fact_bioactivity WHERE ppub_date_key != 19000101
        UNION ALL
        SELECT 'Publication', COUNT(DISTINCT publication_date_key) FROM fact_bioactivity WHERE publication_date_key != 19000101;
        """

        integrity_query = """
        WITH fact_unique_dates AS (
            SELECT DISTINCT date_key FROM (
                SELECT received_date_key as date_key FROM fact_bioactivity WHERE received_date_key != 19000101
                UNION SELECT revised_date_key FROM fact_bioactivity WHERE revised_date_key != 19000101
                UNION SELECT accepted_date_key FROM fact_bioactivity WHERE accepted_date_key != 19000101
                UNION SELECT epub_date_key FROM fact_bioactivity WHERE epub_date_key != 19000101
                UNION SELECT ppub_date_key FROM fact_bioactivity WHERE ppub_date_key != 19000101
                UNION SELECT publication_date_key FROM fact_bioactivity WHERE publication_date_key != 19000101
            ) all_facts
        )
        SELECT 
            (SELECT COUNT(*) FROM fact_unique_dates) as distinct_dates_in_facts,
            (SELECT COUNT(*) FROM dim_date WHERE date_key != 19000101) as total_dates_in_dim,
            (SELECT COUNT(*) FROM fact_unique_dates WHERE date_key NOT IN (SELECT date_key FROM dim_date)) as facts_missing_in_dim,
            (SELECT COUNT(*) FROM dim_date WHERE date_key != 19000101 AND date_key NOT IN (SELECT date_key FROM fact_unique_dates)) as dim_missing_in_facts;
        """

        macro_compression_query = """
        SELECT 
            (SELECT COUNT(*) FROM fact_bioactivity WHERE received_date_key != 19000101) +
            (SELECT COUNT(*) FROM fact_bioactivity WHERE revised_date_key != 19000101) +
            (SELECT COUNT(*) FROM fact_bioactivity WHERE accepted_date_key != 19000101) +
            (SELECT COUNT(*) FROM fact_bioactivity WHERE epub_date_key != 19000101) +
            (SELECT COUNT(*) FROM fact_bioactivity WHERE ppub_date_key != 19000101) +
            (SELECT COUNT(*) FROM fact_bioactivity WHERE publication_date_key != 19000101) as total_fk_references_in_facts,
            (SELECT COUNT(*) FROM dim_date WHERE date_key != 19000101) as total_physical_rows_in_dim;
        """

        try:
            with self.db_manager.get_dwh_connection() as conn:
                df_stats = pd.read_sql(stats_query, conn)
                df_integrity = pd.read_sql(integrity_query, conn)
                df_macro = pd.read_sql(macro_compression_query, conn)

            print("\nDATA QUALITY & ARCHITECTURE REPORT")

            print("\n1. Unique Dates by Type in Fact Table")
            print(df_stats.to_string(index=False))
            
            print("\n2. Referential Integrity Check")
            print(df_integrity.to_string(index=False))
            
            print("\n3. Macro Storage Compression")
            print(df_macro.to_string(index=False))
            
            print("\nFINAL CONCLUSIONS")
            
            if df_integrity.loc[0, 'distinct_dates_in_facts'] == df_integrity.loc[0, 'total_dates_in_dim']:
                print("INTEGRITY: Dates in Facts exactly match the Dimension table.")
            if df_integrity.loc[0, 'facts_missing_in_dim'] == 0:
                print("COVERAGE: 100% Coverage. No orphan dates in Facts.")
            if df_integrity.loc[0, 'dim_missing_in_facts'] == 0:
                print("EFFICIENCY: Zero Excess. No unused dates in Dimension.")

            print(f"\nARCHITECTURE: {df_macro.loc[0, 'total_physical_rows_in_dim']} physical rows support {df_macro.loc[0, 'total_fk_references_in_facts']:,} references.")

        except Exception as e:
            logger.error("Error executing report: %s", e)