import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import base64
from typing import List, Dict, Tuple, Optional
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class GitHubReadmeClusterer:
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the GitHub README clusterer.
        
        Args:
            github_token: GitHub personal access token for API requests
        """
        self.github_token = github_token
        self.headers = {'Authorization': f'token {github_token}'} if github_token else {}
        self.repositories = []
        self.readme_texts = []
        self.features = None
        self.clusters = None
        self.vectorizer = None
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def search_repositories(self, query: str, max_repos: int = 100) -> List[Dict]:
        """
        Search for repositories using GitHub API.
        
        Args:
            query: Search query (e.g., 'machine learning', 'web development')
            max_repos: Maximum number of repositories to fetch
            
        Returns:
            List of repository information dictionaries
        """
        repositories = []
        page = 1
        per_page = min(100, max_repos)
        
        while len(repositories) < max_repos:
            url = f"https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'page': page,
                'per_page': per_page
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                repositories.extend(data['items'])
                
                if len(data['items']) < per_page:
                    break
                    
                page += 1
                time.sleep(1)  # Rate limiting
            else:
                print(f"Error fetching repositories: {response.status_code}")
                break
        
        return repositories[:max_repos]
    
    def fetch_readme(self, repo_owner: str, repo_name: str) -> Optional[str]:
        """
        Fetch README content from a repository.
        
        Args:
            repo_owner: Repository owner username
            repo_name: Repository name
            
        Returns:
            README content as string or None if not found
        """
        readme_files = ['README.md', 'readme.md', 'README.rst', 'README.txt', 'README']
        
        for readme_file in readme_files:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{readme_file}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                content = response.json()
                if content['encoding'] == 'base64':
                    readme_content = base64.b64decode(content['content']).decode('utf-8')
                    return readme_content
            
            time.sleep(0.5)  # Rate limiting
        
        return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess README text for better clustering.
        
        Args:
            text: Raw README text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks
        text = re.sub(r'`[^`]*`', '', text)  # Inline code
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Links
        text = re.sub(r'#+\s*', '', text)  # Headers
        text = re.sub(r'\*\*|__', '', text)  # Bold
        text = re.sub(r'\*|_', '', text)  # Italic
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def collect_data(self, queries: List[str], repos_per_query: int = 50):
        """
        Collect README data from multiple search queries.
        
        Args:
            queries: List of search queries
            repos_per_query: Number of repositories per query
        """
        print("Collecting repository data...")
        all_repos = []
        
        for query in queries:
            print(f"Searching for: {query}")
            repos = self.search_repositories(query, repos_per_query)
            all_repos.extend(repos)
        
        # Remove duplicates based on full_name
        seen = set()
        unique_repos = []
        for repo in all_repos:
            if repo['full_name'] not in seen:
                seen.add(repo['full_name'])
                unique_repos.append(repo)
        
        print(f"Found {len(unique_repos)} unique repositories")
        
        # Fetch README files
        successful_repos = []
        readme_texts = []
        
        for i, repo in enumerate(unique_repos):
            print(f"Fetching README {i+1}/{len(unique_repos)}: {repo['full_name']}")
            
            readme_content = self.fetch_readme(repo['owner']['login'], repo['name'])
            
            if readme_content:
                processed_text = self.preprocess_text(readme_content)
                if len(processed_text.strip()) > 50:  # Filter out very short READMEs
                    successful_repos.append(repo)
                    readme_texts.append(processed_text)
        
        self.repositories = successful_repos
        self.readme_texts = readme_texts
        
        print(f"Successfully collected {len(self.repositories)} READMEs")
    
    def extract_features(self, max_features: int = 1000, min_df: int = 2, max_df: float = 0.8):
        """
        Extract TF-IDF features from README texts.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        print("Extracting TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),  # Include bigrams
            stop_words='english'
        )
        
        self.features = self.vectorizer.fit_transform(self.readme_texts)
        print(f"Feature matrix shape: {self.features.shape}")
    
    def find_optimal_clusters(self, max_clusters: int = 20) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features, cluster_labels))
        
        # Find elbow point
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_point = np.argmax(diffs2) + 2
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow: {elbow_point}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(k_range, silhouette_scores, 'go-')
        plt.axvline(x=best_silhouette_k, color='r', linestyle='--', label=f'Best: {best_silhouette_k}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Choose the better of the two methods
        optimal_k = best_silhouette_k if silhouette_scores[best_silhouette_k - 2] > 0.3 else elbow_point
        print(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters: Optional[int] = None, method: str = 'kmeans'):
        """
        Perform clustering on the README features.
        
        Args:
            n_clusters: Number of clusters (if None, will find optimal)
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"Performing {method} clustering with {n_clusters} clusters...")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = clusterer.fit_predict(self.features)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            self.clusters = clusterer.fit_predict(self.features.toarray())
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.clusters = clusterer.fit_predict(self.features.toarray())
        
        print(f"Clustering completed. Found {len(set(self.clusters))} clusters")
        
        # Calculate silhouette score
        if len(set(self.clusters)) > 1:
            sil_score = silhouette_score(self.features, self.clusters)
            print(f"Silhouette Score: {sil_score:.3f}")
    
    def visualize_clusters(self):
        """Visualize clusters using PCA reduction."""
        print("Creating cluster visualization...")
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.features.toarray())
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.clusters, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Repository Clusters (PCA Visualization)')
        
        # Add cluster centers for KMeans
        if hasattr(self, 'clusterer') and hasattr(self.clusterer, 'cluster_centers_'):
            centers_2d = pca.transform(self.clusterer.cluster_centers_)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3)
        
        plt.show()
    
    def analyze_clusters(self):
        """Analyze and display cluster characteristics."""
        print("Analyzing clusters...")
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame({
            'repo_name': [repo['full_name'] for repo in self.repositories],
            'description': [repo.get('description', '') for repo in self.repositories],
            'stars': [repo['stargazers_count'] for repo in self.repositories],
            'language': [repo.get('language', 'Unknown') for repo in self.repositories],
            'cluster': self.clusters,
            'readme_text': self.readme_texts
        })
        
        # Cluster statistics
        cluster_stats = df.groupby('cluster').agg({
            'repo_name': 'count',
            'stars': ['mean', 'median', 'max'],
            'language': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }).round(2)
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        # Top terms per cluster
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in sorted(set(self.clusters)):
            if cluster_id == -1:  # Skip noise cluster for DBSCAN
                continue
                
            print(f"\n=== Cluster {cluster_id} ===")
            cluster_repos = df[df['cluster'] == cluster_id]
            
            print(f"Size: {len(cluster_repos)} repositories")
            print(f"Average stars: {cluster_repos['stars'].mean():.0f}")
            print(f"Most common language: {cluster_repos['language'].mode().iloc[0]}")
            
            # Top repositories in cluster
            top_repos = cluster_repos.nlargest(3, 'stars')
            print("Top repositories:")
            for _, repo in top_repos.iterrows():
                print(f"  - {repo['repo_name']} ({repo['stars']} stars)")
            
            # Extract top terms for this cluster
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_features = self.features[cluster_indices].mean(axis=0).A1
            top_terms_idx = cluster_features.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_terms_idx]
            print(f"Key terms: {', '.join(top_terms)}")
    
    def generate_cluster_wordclouds(self):
        """Generate word clouds for each cluster."""
        print("Generating word clouds...")
        
        unique_clusters = sorted(set(self.clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise cluster
        
        n_clusters = len(unique_clusters)
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_texts = [self.readme_texts[j] for j in range(len(self.readme_texts)) 
                           if self.clusters[j] == cluster_id]
            combined_text = ' '.join(cluster_texts)
            
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white', 
                                max_words=50).generate(combined_text)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Cluster {cluster_id}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: str = 'clustering_results.csv'):
        """Save clustering results to CSV file."""
        df = pd.DataFrame({
            'repository': [repo['full_name'] for repo in self.repositories],
            'description': [repo.get('description', '') for repo in self.repositories],
            'stars': [repo['stargazers_count'] for repo in self.repositories],
            'language': [repo.get('language', 'Unknown') for repo in self.repositories],
            'url': [repo['html_url'] for repo in self.repositories],
            'cluster': self.clusters
        })
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return df

# Example usage
def main():
    # Initialize clusterer (add your GitHub token for higher rate limits)
    clusterer = GitHubReadmeClusterer(github_token=None)  # Replace with your token
    
    # Define search queries for different types of projects
    search_queries = [
        'machine learning python',
        'web development javascript',
        'data science',
        'mobile app development',
        'deep learning tensorflow',
        'react frontend',
        'backend api',
        'devops docker'
    ]
    
    # Collect data
    clusterer.collect_data(search_queries, repos_per_query=20)
    
    # Extract features
    clusterer.extract_features(max_features=500)
    
    # Perform clustering
    clusterer.perform_clustering(method='kmeans')
    
    # Visualize and analyze results
    clusterer.visualize_clusters()
    clusterer.analyze_clusters()
    clusterer.generate_cluster_wordclouds()
    
    # Save results
    results_df = clusterer.save_results()
    
    return clusterer, results_df

if __name__ == "__main__":
    clusterer, results = main()