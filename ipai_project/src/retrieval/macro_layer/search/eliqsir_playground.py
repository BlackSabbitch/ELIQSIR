import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.display import HTML as IHTML

class ELIQSIRPlayground:
    def __init__(self, lexical_retriever, neural_indexer, df_metadata):
        self.lexical = lexical_retriever
        self.neural = neural_indexer
        self.df = df_metadata.copy()
        self.df['article_key'] = self.df['article_key'].astype(int)
        self.df = self.df.set_index('article_key')
        
        self.model_dropdown = widgets.Dropdown(
            options=[
                ('Boolean Search (Exact)', 'boolean'),
                ('TF-IDF (VSM Cosine)', 'tfidf'),
                ('BM25 (Okapi)', 'bm25'), 
                ('Neural (Semantic)', 'neural'),
                ('Hybrid (RRF)', 'hybrid')
            ],
            value='hybrid',
            description='Model:',
            style={'description_width': 'initial'}
        )

        self.query_input = widgets.Text(
            value='Inhibition of phosphodiesterases',
            description='Query:',
            layout=widgets.Layout(width='50%')
        )

        self.top_k_slider = widgets.IntSlider(
            value=5, min=1, max=10, step=1,
            description='Top K:',
        )

        self.search_button = widgets.Button(
            description='Run Search',
            button_style='success',
            layout=widgets.Layout(width='200px'),
            icon='search'
        )
        
        self.output_area = widgets.Output()
        self.search_button.on_click(self._on_search_clicked)

    def _hybrid_search_logic(self, query, top_k):
        candidate_k = max(60, top_k * 2)
        bm25_res = self.lexical.search_bm25(query, top_k=candidate_k)
        neural_res = self.neural.search(query, top_k=candidate_k)
        
        scores = {}
        for rank, res in enumerate(bm25_res, 1):
            key = int(res['article_key'])
            scores[key] = scores.get(key, 0) + (1.0 / (60 + rank))
        for rank, res in enumerate(neural_res, 1):
            key = int(res['article_key'])
            scores[key] = scores.get(key, 0) + (1.0 / (60 + rank))
            
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'article_key': k, 'score': s} for k, s in sorted_keys[:top_k]]

    def _on_search_clicked(self, b):
        with self.output_area:
            clear_output()
            query = self.query_input.value
            model = self.model_dropdown.value
            k = self.top_k_slider.value
            
            try:
                if model == 'boolean':
                    results = self.lexical.search_boolean(query)[:k] 
                elif model == 'tfidf':
                    results = self.lexical.search_vsm(query, top_k=k)
                elif model == 'bm25':
                    results = self.lexical.search_bm25(query, top_k=k)
                elif model == 'neural':
                    results = self.neural.search(query, top_k=k)
                else:
                    results = self._hybrid_search_logic(query, k)
                    
                self._render_results(results)
            except Exception as e:
                print(f"Search Error: {e}")

    def _render_results(self, results):
        if not results:
            print("No documents matched.")
            return

        html = f"<h3>Found {len(results)} matches:</h3><br>"
        
        for i, res in enumerate(results, 1):
            key = int(res['article_key'])
            search_score = res.get('score', 0)
            
            try:
                row = self.df.loc[key]
                meta = row.get('metadata', {})
                title = meta.get('title', 'Untitled')
                authors = meta.get('authors', 'Unknown Authors')
                journal = meta.get('journal', 'Unknown Journal')
                year = meta.get('year', 'N/A')
                
                metrics = row.get('metrics', {})
                pchembl = metrics.get('avg_pchembl', 'N/A')
                
                full_text = row.get('search_corpus', '')
                snippet = (full_text[:350] + '...') if len(full_text) > 350 else full_text
            except (KeyError, TypeError):
                title, authors, journal, year, pchembl, snippet = f"Article {key}", "N/A", "N/A", "N/A", "N/A", "Content missing in JSONL."

            html += f"""
            <div style="border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; 
                        background-color: #fcfcfc; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); font-family: sans-serif;">
                <div style="color: #1a0dab; font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                    {i}. {title}
                </div>
                <div style="color: #545454; font-size: 13px; margin-bottom: 8px;">
                    <b>Authors:</b> {authors} | <b>Journal:</b> {journal} ({year})
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="background-color: #e8f0fe; color: #1967d2; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                        Article Key: {key}
                    </span>
                    <span style="background-color: #e6f4ea; color: #137333; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-left: 5px;">
                        Search Score: {search_score:.4f}
                    </span>
                    <span style="background-color: #fef7e0; color: #b06000; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-left: 5px;">
                        Avg pChEMBL: {pchembl}
                    </span>
                </div>
                <div style="color: #3c4043; font-size: 14px; line-height: 1.5; text-align: justify;">
                    {snippet}
                </div>
            </div>
            """
        
        display(IHTML(html))

    def show(self):
        header = widgets.HTML("<h2>ELIQSIR Search Lab</h2><p>Compare lexical and neural models on scientific corpus.</p>")
        ui = widgets.VBox([
            header,
            self.model_dropdown,
            widgets.HBox([self.query_input, self.top_k_slider]),
            self.search_button,
            self.output_area
        ])
        display(ui)