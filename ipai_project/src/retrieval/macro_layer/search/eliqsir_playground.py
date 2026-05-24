import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.display import HTML as IHTML

class ELIQSIRPlayground:
    def __init__(self, lexical_retriever, neural_indexer, hybrid_retriever, df_metadata):
        self.lexical = lexical_retriever
        self.neural = neural_indexer
        self.hybrid = hybrid_retriever 
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

        # Set default value to 20 and max to 50 for optimal retrieval exploration
        self.top_k_slider = widgets.IntSlider(
            value=20, min=1, max=50, step=1,
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
                    results = self.hybrid.search(query, top_k=k)
                    
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
                
                # Extract keys for hyperlinking ---
                natural_keys = row.get('natural_keys', {})
                pubmed_id = natural_keys.get('pubmed_id')
                doi = natural_keys.get('doi')
                
                # Handle possible list format just in case
                if isinstance(pubmed_id, list) and pubmed_id: pubmed_id = pubmed_id[0]
                if isinstance(doi, list) and doi: doi = doi[0]
                
                # Create a clickable title if an ID exists
                title_html = title
                if pubmed_id:
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                    title_html = f"<a href='{link}' target='_blank' style='color: #1a0dab; text-decoration: underline;'>{title}</a>"
                elif doi:
                    link = f"https://doi.org/{doi}"
                    title_html = f"<a href='{link}' target='_blank' style='color: #1a0dab; text-decoration: underline;'>{title}</a>"
                else:
                    # Fallback for articles without external links
                    title_html = f"<span style='color: #1a0dab;'>{title}</span>"
                  
                full_text = row.get('search_corpus', '')
                snippet = (full_text[:350] + '...') if len(full_text) > 350 else full_text
            except (KeyError, TypeError):
                title_html = f"<span style='color: #1a0dab;'>Article {key}</span>"
                authors, journal, year, pchembl, snippet = "N/A", "N/A", "N/A", "N/A", "Content missing in JSONL."

            html += f"""
            <div style="border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; 
                        background-color: #fcfcfc; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); font-family: sans-serif;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                    {i}. {title_html}
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
        # Display the search lab UI components
        header = widgets.HTML("<h2>ELIQSIR Search Lab</h2><p>Compare lexical, neural and hybrid models on scientific corpus.</p>")
        ui = widgets.VBox([
            header,
            self.model_dropdown,
            widgets.HBox([self.query_input, self.top_k_slider]),
            self.search_button,
            self.output_area
        ])
        display(ui)