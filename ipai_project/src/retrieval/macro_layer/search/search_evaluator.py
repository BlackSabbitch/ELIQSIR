import json
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

    def plot_benchmark_results(self, results_df: pd.DataFrame, metrics: list = None):
        """
        Generates comparative bar charts for multiple metrics.
        If no metrics are provided, it plots all available metric columns.
        """
       
        if metrics is None:
            # Automatically find all metric columns (exclude ID and method)
            valid_metrics = [col for col in results_df.columns if col not in ['Query ID', 'method']]
        else:
            valid_metrics = [m for m in metrics if m in results_df.columns]
            
        if not valid_metrics:
            print("Error: No valid metrics found in the DataFrame.")
            return

        num_metrics = len(valid_metrics)
        # Dynamically adjust height based on the number of metrics (e.g., 5 metrics * 4.5 inches)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 4.5 * num_metrics), sharex=True)
        sns.set_style("whitegrid")
        
        # If there's only one metric, axes is not a list, so we wrap it
        if num_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(valid_metrics):
            ax = sns.barplot(
                data=results_df, 
                x="Query ID", 
                y=metric, 
                hue="method", 
                palette="viridis",
                ax=axes[i]
            )
            
            # Add titles and labels for each subplot
            axes[i].set_title(f"Search Performance: {metric}", fontsize=16, pad=10)
            axes[i].set_ylabel("Score", fontsize=12)
            axes[i].set_ylim(0, 1.1) 
            
            # Display the legend only on the first plot to avoid clutter
            if i == 0:
                axes[i].legend(
                    title="Search Algorithm", 
                    bbox_to_anchor=(1.02, 1), 
                    loc='upper left',
                    fontsize=11,
                    title_fontsize=12
                )
            else:
                if axes[i].get_legend() is not None:
                    axes[i].get_legend().remove()

        plt.xlabel("Test Queries (Q1 - Q10)", fontsize=14)
        plt.tight_layout()
        plt.show()