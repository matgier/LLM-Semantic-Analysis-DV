import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TextAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the analyzer with a specific model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        print("Initializing model...")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = None
        self.results = {}
        
    def process_texts(self, texts):
        """
        Process list of texts through embeddings and dimensionality reduction.
        
        Args:
            texts (list): List of strings to process
            
        Returns:
            dict: Dictionary containing the results of all dimension reductions
        """
        print("Processing texts...")
        self.texts = texts
        self.embeddings = self.model.encode(texts)
        
        # Perform dimensionality reduction
        print("Performing dimensionality reduction...")
        self.results['pca'] = self._run_pca()
        self.results['tsne'] = self._run_tsne()
        self.results['mds'] = self._run_mds()

        return self.results
        
    def _run_pca(self):
        """Run PCA dimensionality reduction."""
        pca = PCA(n_components=3, whiten=True)
        return pca.fit_transform(self.embeddings)
    
    def _run_tsne(self):
        """Run t-SNE dimensionality reduction with dynamic perplexity."""
        n_samples = len(self.texts)
        perplexity = min(40, n_samples - 1)
        print(f"Using perplexity value of {perplexity} for {n_samples} samples")
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=1000
        )
        return tsne.fit_transform(self.embeddings)
    
    def _run_mds(self):
        """Run MDS dimensionality reduction."""
        mds = MDS(n_components=3,  random_state=42, metric=False)
        return mds.fit_transform(self.embeddings)
    
    def get_results_json(self, method='all'):
        """
        Get dimension reduction results in JSON format.
        
        Args:
            method (str): Which method to get results for ('pca', 'tsne', 'mds', or 'all')
            
        Returns:
            str: JSON string containing the requested results
        """
        if not self.results:
            raise ValueError("No results available. Call process_texts() first.")
            
        result_dict = {}
        
        if method == 'all':
            # Return all results
            for method_name, coords in self.results.items():
                result_dict[method_name] = self._coords_to_dict(coords, method_name)
        elif method in self.results:
            # Return specific method
            result_dict = self._coords_to_dict(self.results[method], method)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'tsne', 'mds', or 'all'")
            
        return json.dumps(result_dict, indent=2)
    
    def _coords_to_dict(self, coords, method_name):
        """
        Convert coordinates to a dictionary format suitable for JSON conversion.
        
        Args:
            coords (numpy.ndarray): Coordinates from dimension reduction
            method_name (str): Name of the dimension reduction method
            
        Returns:
            dict: Dictionary containing the coordinates with text IDs
        """
        result = {
            "method": method_name,
            "dimensions": coords.shape[1],
            "points": []
        }
        
        for i, coord in enumerate(coords):
            point = {
                "id": i,
                "text": self.texts[i],
                "coordinates": coord.tolist()
            }
            result["points"].append(point)
            
        return result
    
    def define_context_space(self, method='pca', n_components=2, threshold=0.7):
        """
        Define a context space using PCA or MDS results.
        
        Args:
            method (str): Which method to use ('pca' or 'mds')
            n_components (int): Number of components to use (default 2)
            threshold (float): Similarity threshold for considering texts in same context
            
        Returns:
            dict: Dictionary containing context clusters and related information
        """
        if method not in ['pca', 'mds']:
            raise ValueError("Method must be either 'pca' or 'mds'")
            
        if method not in self.results:
            raise ValueError(f"No {method.upper()} results available. Call process_texts() first.")
            
        # Use only the specified number of components
        coords = self.results[method][:, :n_components]
        
        # Calculate pairwise similarities in the reduced space
        similarities = cosine_similarity(coords)
        
        # Create context clusters
        clusters = {}
        assigned = set()
        
        for i in range(len(self.texts)):
            if i in assigned:
                continue
                
            # Find all texts similar to this one
            similar_indices = np.where(similarities[i] >= threshold)[0]
            
            if len(similar_indices) > 1:  # At least the text itself and one other
                cluster_id = len(clusters)
                clusters[cluster_id] = {
                    "center_text_id": i,
                    "center_text": self.texts[i],
                    "members": [{"id": j, "text": self.texts[j], "similarity": similarities[i, j]} 
                               for j in similar_indices]
                }
                assigned.update(similar_indices)
        
        # Add singleton clusters for unassigned texts
        for i in range(len(self.texts)):
            if i not in assigned:
                cluster_id = len(clusters)
                clusters[cluster_id] = {
                    "center_text_id": i,
                    "center_text": self.texts[i],
                    "members": [{"id": i, "text": self.texts[i], "similarity": 1.0}]
                }
        
        context_space = {
            "method": method,
            "n_components": n_components,
            "threshold": threshold,
            "n_clusters": len(clusters),
            "clusters": clusters
        }
        
        return context_space
    
    def save_context_space_json(self, context_space, output_file='context_space.json'):
        """
        Save context space to a JSON file.
        
        Args:
            context_space (dict): Context space dictionary from define_context_space()
            output_file (str): Path to save the JSON output
        """
        with open(output_file, 'w') as f:
            json.dump(context_space, f, indent=2)
        print(f"Context space saved to {output_file}")
    
    def _get_colors(self, color_by='sequence'):
        """
        Get colors for points based on specified method.
        
        Args:
            color_by (str): Method to determine colors ('sequence', 'similarity', or 'cluster')
        
        Returns:
            tuple: (colors array, colorscale name, color title)
        """
        if color_by == 'sequence':
            colors = np.arange(len(self.texts))
            colorscale = 'viridis'  # Changed from Viridis
            color_title = 'Sequence'
        
        elif color_by == 'similarity':
            # Kolorowanie bazujące na podobieństwie do pierwszego tekstu
            similarities = np.array([
                cosine_similarity(
                    self.embeddings[0].reshape(1, -1),
                    text_embed.reshape(1, -1)
                )[0][0]
                for text_embed in self.embeddings
            ])
            colors = similarities
            colorscale = 'RdBu'  # Changed from RdYlBu
            color_title = 'Similarity to first text'
        
        elif color_by == 'cluster':
            # Kolorowanie bazujące na klastrach
            n_clusters = min(5, len(self.texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            colors = kmeans.fit_predict(self.embeddings)
            colorscale = 'plotly3'  # Changed from Set1
            color_title = 'Cluster'
        
        else:
            raise ValueError(f"Unknown color_by method: {color_by}")
            
        return colors, colorscale, color_title
    
    def create_visualization(self, output_file='dimension_reduction.html', color_by='sequence'):
        """
        Create interactive visualization using plotly.
        
        Args:
            output_file (str): Path to save the HTML output
            color_by (str): How to color points ('sequence', 'similarity', or 'cluster')
        """
        print(f"Creating visualization at {output_file}...")
        
        # Get colors for points
        colors, colorscale, color_title = self._get_colors(color_by)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('PCA Visualization', 't-SNE Visualization', 'MDS Visualization'),
            horizontal_spacing=0.1
        )
        
        # Add traces for each method
        for idx, (method, coords) in enumerate(self.results.items(), 1):
            # Normalize coordinates
            coords_norm = coords.copy()
            if len(coords) > 1:
                for i in range(coords.shape[1]):
                    min_val = coords[:, i].min()
                    max_val = coords[:, i].max()
                    if max_val > min_val:
                        coords_norm[:, i] = (coords[:, i] - min_val) / (max_val - min_val)
            
            hover_text = [
                f"Text {i+1}<br>Original: {text}<br>{color_title}: {colors[i]:.2f}"
                for i, text in enumerate(self.texts)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=coords_norm[:, 0],
                    y=coords_norm[:, 1],
                    mode='markers+text',
                    name=method.upper(),
                    text=[f' {i+1}' for i in range(len(self.texts))],
                    hovertext=hover_text,
                    hoverinfo='text',
                    customdata=self.texts,
                    marker=dict(
                        size=10,
                        color=colors,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title=color_title)
                    )
                ),
                row=1,
                col=idx
            )

        # Update layout
        fig.update_layout(
            title_text=f"Interactive Text Visualization (colored by {color_title})",
            showlegend=True,
            height=800,
            width=2000,
            template='plotly_white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(title_text="First Component", row=1, col=i, range=[-0.1, 1.1])
            fig.update_yaxes(title_text="Second Component", row=1, col=i, range=[-0.1, 1.1])

        # Save to HTML file
        fig.write_html(
            output_file,
            include_plotlyjs=True,
            full_html=True,
        )
        
        return fig