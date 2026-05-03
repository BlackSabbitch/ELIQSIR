import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.retrieval.macro_layer.search.lexical_retriever import LexicalRetriever
from src.retrieval.macro_layer.search.ground_truth_manager import GroundTruthManager
from src.retrieval.macro_layer.search.metrics_engine import MetricsEngine

class SearchEvaluator:
    """
    Orchestrates the evaluation process: displays snippets for manual review,
    calculates final benchmark metrics using the MetricsEngine, and visualizes results.
    """
    def __init__(self, retriever: LexicalRetriever, gt_manager: GroundTruthManager):
        self.retriever = retriever
        self.gt = gt_manager
        self.snippets = self._load_snippets()

    def _load_snippets(self) -> dict:
        # Load corpus into memory for fast snippet retrieval
        snippets = {}
        if not self.gt.corpus_path.exists():
            print(f"WARNING: Corpus not found at {self.gt.corpus_path}")
            return snippets
            
        with open(self.gt.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                snippets[str(doc['article_key'])] = doc.get('search_corpus', '')[:300] + "..."
        return snippets

    def inspect_results(self, top_k=5):
        # Displays top results from BM25 for manual sanity checks
        print("MANUAL INSPECTION MODE (BUILDING GROUND TRUTH)")
        
        for qid, query in self.gt.queries.items():
            print(f"\n[ {qid} ] QUERY: '{query}'\n{'-'*40}")
            
            results = self.retriever.search_bm25(query, top_k=top_k)
            
            if not results:
                print("  No results found.")
                continue
            
            for rank, res in enumerate(results, 1):
                key = res['article_key']
                text = self.snippets.get(key, "No text found.")
                print(f"#{rank} | Key: '{key}' | Score: {res['score']}")
                print(f"Text: {text}\n")

    def evaluate_model(self, model_name: str, search_callable, k: int = 5) -> pd.DataFrame:
        # Calculates all fundamental IR metrics for a given model
        print(f"\nEVALUATING MODEL: {model_name} @ K={k}")
        report = []
        
        for qid, query in self.gt.queries.items():
            relevant_docs = set(self.gt.qrels.get(qid, []))
            
            if not relevant_docs:
                continue
                
            # Handle Boolean Search explicitly since it doesn't take top_k
            if "Boolean" in model_name:
                raw_results = search_callable(query) 
            else:
                raw_results = search_callable(query, top_k=k)
            
            retrieved_docs = [str(r['article_key']) for r in raw_results][:k]
            
            report.append({
                "Query ID": qid,
                f"P@{k}": round(MetricsEngine.precision_at_k(relevant_docs, retrieved_docs, k), 3),
                f"Recall@{k}": round(MetricsEngine.recall(relevant_docs, retrieved_docs), 3),
                f"F1@{k}": round(MetricsEngine.f1_score(relevant_docs, retrieved_docs), 3),
                f"nDCG@{k}": round(MetricsEngine.ndcg_at_k(relevant_docs, retrieved_docs, k), 3),
                f"AP@{k}": round(MetricsEngine.average_precision(relevant_docs, retrieved_docs), 3)
            })
            
        df = pd.DataFrame(report)
        if df.empty:
            print("No queries evaluated. Have you annotated the Ground Truth yet?")
            return df
            
        print(df.drop(columns=[f'AP@{k}']).to_string(index=False))
        
        print(f"\nMean Average Metrics ({model_name}):")
        summary = df.mean(numeric_only=True).round(3).to_dict()
        summary[f"MAP@{k}"] = summary.pop(f"AP@{k}")
        
        for metric, val in summary.items():
            print(f"  {metric}: {val}")
            
        return df

    def plot_benchmark_results(self, results_df: pd.DataFrame, metrics: list = None, top_q: int = 5):
        # Generates comparative bar charts for multiple metrics.
        unique_qids = results_df['Query ID'].unique()[:top_q]
        plot_df = results_df[results_df['Query ID'].isin(unique_qids)]
        
        if metrics is None:
            valid_metrics = [col for col in plot_df.columns if col not in ['Query ID', 'method']]
        else:
            valid_metrics = [m for m in metrics if m in plot_df.columns]
            
        if not valid_metrics:
            print("Error: No valid metrics found in the DataFrame.")
            return

        num_metrics = len(valid_metrics)
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 4.5 * num_metrics), sharex=False)
        sns.set_style("whitegrid")
        
        if num_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(valid_metrics):
            ax = sns.barplot(
                data=plot_df, 
                x="Query ID", 
                y=metric, 
                hue="method", 
                palette="viridis",
                ax=axes[i]
            )
            
            axes[i].set_title(f"Search Performance: {metric}", fontsize=16, pad=10)
            axes[i].set_ylabel("Score", fontsize=12)
            axes[i].set_ylim(0, 1.1) 
            axes[i].set_xlabel("Test Queries", fontsize=12) 
            
            if i == 0:
                axes[i].legend(
                    title="Search Algorithm", 
                    bbox_to_anchor=(1.02, 1), 
                    loc='upper left',
                    fontsize=11,
                    title_fontsize=12
                )
                
                import textwrap
                query_lines = []
                for qid in unique_qids:
                    q_text = self.gt.queries.get(qid, 'Unknown')
                    wrapped_text = textwrap.fill(f"{qid}: {q_text}", width=40)
                    query_lines.append(wrapped_text)
                
                query_mapping_str = "Query Mapping:\n" + "-"*30 + "\n" + "\n\n".join(query_lines)
                
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
                axes[i].text(1.02, 0.5, query_mapping_str, 
                             transform=axes[i].transAxes, fontsize=10,
                             verticalalignment='top', bbox=props)
            else:
                if axes[i].get_legend() is not None:
                    axes[i].get_legend().remove()

        plt.tight_layout()
        plt.show()

    def export_pool_for_ai_review(self, neural_indexer, target_qids: list, top_k: int = 5):
        """
        Gathers top-K candidates from both BM25 and Neural Indexer,
        shuffles them, and prints them in a format ready for AI assessment.
        """
        print("=== DATA EXPORT FOR AI ASSESSOR ===\n")
        
        for qid in target_qids:
            query = self.gt.queries.get(qid)
            if not query:
                continue
                
            print(f"==================================================")
            print(f"QUERY [{qid}]: {query}")
            print(f"==================================================")
            
            # Fetch candidates from both search methods
            bm25_res = [str(res['article_key']) for res in self.retriever.search_bm25(query, top_k=top_k)]
            neural_res = [str(res['article_key']) for res in neural_indexer.search(query, top_k=top_k)]
            
            # Merge and remove duplicates
            pool = list(set(bm25_res + neural_res))
            random.shuffle(pool)
            
            for doc_id in pool:
                text = self.snippets.get(doc_id, "Text not found.")[:500]
                text = text.replace('\n', ' ').strip()
                print(f"ID: {doc_id} | Text: {text}...\n")
                
        print("=== END OF EXPORT ===")

    def evaluate_bm25_vs_neural(self, neural_indexer, k: int = 10, metrics_to_plot: list = None) -> pd.DataFrame:
        """
        Runs the evaluation pipeline for both BM25 and Neural search models,
        combines the data into a single DataFrame, and plots the results.
        """
        print("\n" + "="*50)
        print("RUNNING FULL A/B BENCHMARK")
        print("="*50)
        
        # 1. Evaluate BM25
        df_bm25 = self.evaluate_model("BM25", self.retriever.search_bm25, k=k)
        df_bm25['method'] = 'BM25' 
        
        # 2. Evaluate Neural Indexer
        df_neural = self.evaluate_model("Neural (MiniLM)", neural_indexer.search, k=k)
        df_neural['method'] = 'Neural (Semantic)' 
        
        # 3. Combine DataFrames
        df_combined = pd.concat([df_bm25, df_neural], ignore_index=True)
        
        # 4. Generate the comparative plots
        print("\nGenerating Benchmark Plots...")
        self.plot_benchmark_results(df_combined, metrics=metrics_to_plot)
        
        return df_combined