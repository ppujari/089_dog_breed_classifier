import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import base64
from typing import List, Dict, Tuple, Optional, Union
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
warnings.filterwarnings('ignore')

class GitHubReadmeOutlierDetector:
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the GitHub README outlier detector.
        
        Args:
            github_token: GitHub personal access token for API requests
        """
        self.github_token = github_token
        self.headers = {'Authorization': f'token {github_token}'} if github_token else {}
        self.repositories = []
        self.readme_texts = []
        self.readme_features = None
        self.outlier_scores = {}
        self.outlier_labels = {}
        self.vectorizer = None
        self.feature_names = []
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def search_repositories(self, query: str, max_repos: int = 200) -> List[Dict]:
        """Search for repositories using GitHub API."""
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
                time.sleep(1)
            else:
                print(f"Error fetching repositories: {response.status_code}")
                break
        
        return repositories[:max_repos]
    
    def fetch_readme(self, repo_owner: str, repo_name: str) -> Optional[str]:
        """Fetch README content from a repository."""
        readme_files = ['README.md', 'readme.md', 'README.rst', 'README.txt', 'README']
        
        for readme_file in readme_files:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{readme_file}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                content = response.json()
                if content['encoding'] == 'base64':
                    readme_content = base64.b64decode(content['content']).decode('utf-8')
                    return readme_content
            
            time.sleep(0.5)
        
        return None
    
    def extract_readme_features(self, text: str) -> Dict:
        """
        Extract comprehensive features from README text for outlier detection.
        
        Args:
            text: Raw README text
            
        Returns:
            Dictionary of extracted features
        """
        if not text:
            return {feature: 0 for feature in self.get_feature_names()}
        
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['line_count'] = len(text.split('\n'))
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        # Markdown-specific features
        features['header_count'] = len(re.findall(r'^#+\s', text, re.MULTILINE))
        features['code_block_count'] = len(re.findall(r'```', text)) // 2
        features['inline_code_count'] = len(re.findall(r'`[^`]+`', text))
        features['link_count'] = len(re.findall(r'\[.*?\]\(.*?\)', text))
        features['image_count'] = len(re.findall(r'!\[.*?\]\(.*?\)', text))
        features['list_item_count'] = len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE))
        features['numbered_list_count'] = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
        features['table_count'] = len(re.findall(r'\|.*\|', text))
        features['emoji_count'] = len(re.findall(r':[a-zA-Z_]+:', text))
        features['badge_count'] = len(re.findall(r'!\[.*?\]\(https://img\.shields\.io', text))
        
        # URL and email patterns
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        # Special characters and formatting
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        # Language and technical terms
        tech_terms = ['api', 'framework', 'library', 'database', 'algorithm', 'machine learning', 
                     'artificial intelligence', 'deep learning', 'neural network', 'docker', 
                     'kubernetes', 'microservice', 'frontend', 'backend', 'fullstack']
        
        text_lower = text.lower()
        for term in tech_terms:
            features[f'tech_term_{term.replace(" ", "_")}'] = text_lower.count(term)
        
        # Programming language mentions
        languages = ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 
                    'php', 'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab']
        
        for lang in languages:
            features[f'lang_{lang}'] = text_lower.count(lang)
        
        # Documentation quality indicators
        features['has_installation'] = int('install' in text_lower)
        features['has_usage'] = int('usage' in text_lower or 'example' in text_lower)
        features['has_contributing'] = int('contribut' in text_lower)
        features['has_license'] = int('license' in text_lower)
        features['has_changelog'] = int('changelog' in text_lower or 'change log' in text_lower)
        features['has_testing'] = int('test' in text_lower or 'testing' in text_lower)
        features['has_documentation'] = int('documentation' in text_lower or 'docs' in text_lower)
        
        # Unique patterns
        features['has_ascii_art'] = int(bool(re.search(r'[^\w\s]{10,}', text)))
        features['has_very_long_lines'] = int(any(len(line) > 200 for line in text.split('\n')))
        features['has_repeated_chars'] = int(bool(re.search(r'(.)\1{5,}', text)))
        features['has_multiple_languages'] = int(sum(features[f'lang_{lang}'] for lang in languages) > 3)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        base_features = [
            'length', 'word_count', 'line_count', 'paragraph_count', 'sentence_count',
            'avg_sentence_length', 'avg_word_length', 'flesch_reading_ease', 'flesch_kincaid_grade',
            'header_count', 'code_block_count', 'inline_code_count', 'link_count', 'image_count',
            'list_item_count', 'numbered_list_count', 'table_count', 'emoji_count', 'badge_count',
            'url_count', 'email_count', 'exclamation_count', 'question_count',
            'uppercase_ratio', 'punctuation_ratio', 'digit_ratio'
        ]
        
        tech_terms = ['api', 'framework', 'library', 'database', 'algorithm', 'machine_learning', 
                     'artificial_intelligence', 'deep_learning', 'neural_network', 'docker', 
                     'kubernetes', 'microservice', 'frontend', 'backend', 'fullstack']
        
        languages = ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 
                    'php', 'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab']
        
        quality_features = [
            'has_installation', 'has_usage', 'has_contributing', 'has_license', 
            'has_changelog', 'has_testing', 'has_documentation'
        ]
        
        unique_features = [
            'has_ascii_art', 'has_very_long_lines', 'has_repeated_chars', 'has_multiple_languages'
        ]
        
        all_features = base_features + [f'tech_term_{term}' for term in tech_terms] + \
                      [f'lang_{lang}' for lang in languages] + quality_features + unique_features
        
        return all_features
    
    def collect_data(self, queries: List[str], repos_per_query: int = 50):
        """Collect README data from multiple search queries."""
        print("Collecting repository data...")
        all_repos = []
        
        for query in queries:
            print(f"Searching for: {query}")
            repos = self.search_repositories(query, repos_per_query)
            all_repos.extend(repos)
        
        # Remove duplicates
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
                if len(readme_content.strip()) > 100:  # Filter very short READMEs
                    successful_repos.append(repo)
                    readme_texts.append(readme_content)
        
        self.repositories = successful_repos
        self.readme_texts = readme_texts
        
        print(f"Successfully collected {len(self.repositories)} READMEs")
    
    def extract_features(self):
        """Extract features from all README texts."""
        print("Extracting features from README files...")
        
        # Extract structural and content features
        feature_list = []
        for text in self.readme_texts:
            features = self.extract_readme_features(text)
            feature_list.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_list)
        
        # Add TF-IDF features for content analysis
        print("Adding TF-IDF features...")
        preprocessed_texts = [self.preprocess_text(text) for text in self.readme_texts]
        
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_features = self.vectorizer.fit_transform(preprocessed_texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{feature}' for feature in self.vectorizer.get_feature_names_out()]
        )
        
        # Combine all features
        self.readme_features = pd.concat([feature_df, tfidf_df], axis=1)
        self.readme_features = self.readme_features.fillna(0)
        
        print(f"Total features extracted: {self.readme_features.shape[1]}")
        
        return self.readme_features
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF analysis."""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'#+\s*', '', text)
        
        # Remove URLs and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        
        # Tokenize and clean
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1) -> Dict:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with outlier labels and scores
        """
        print("Detecting outliers using Isolation Forest...")
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(self.readme_features)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(scaled_features)
        outlier_scores = iso_forest.decision_function(scaled_features)
        
        # Convert labels (-1 for outliers, 1 for inliers)
        outlier_labels = (outlier_labels == -1).astype(int)
        
        return {
            'labels': outlier_labels,
            'scores': outlier_scores,
            'method': 'Isolation Forest'
        }
    
    def detect_outliers_lof(self, n_neighbors: int = 20, contamination: float = 0.1) -> Dict:
        """
        Detect outliers using Local Outlier Factor.
        
        Args:
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with outlier labels and scores
        """
        print("Detecting outliers using Local Outlier Factor...")
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(self.readme_features)
        
        # Apply LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(scaled_features)
        outlier_scores = lof.negative_outlier_factor_
        
        # Convert labels
        outlier_labels = (outlier_labels == -1).astype(int)
        
        return {
            'labels': outlier_labels,
            'scores': outlier_scores,
            'method': 'Local Outlier Factor'
        }
    
    def detect_outliers_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """
        Detect outliers using DBSCAN clustering.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a cluster
            
        Returns:
            Dictionary with outlier labels and scores
        """
        print("Detecting outliers using DBSCAN...")
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(self.readme_features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        # Points labeled as -1 are outliers
        outlier_labels = (cluster_labels == -1).astype(int)
        
        # Generate scores based on distance to nearest cluster
        scores = np.zeros(len(scaled_features))
        for i, point in enumerate(scaled_features):
            if outlier_labels[i] == 1:  # If outlier
                # Distance to nearest non-outlier point
                non_outlier_indices = np.where(outlier_labels == 0)[0]
                if len(non_outlier_indices) > 0:
                    distances = np.linalg.norm(scaled_features[non_outlier_indices] - point, axis=1)
                    scores[i] = -np.min(distances)  # Negative for consistency
        
        return {
            'labels': outlier_labels,
            'scores': scores,
            'method': 'DBSCAN'
        }
    
    def detect_outliers_ensemble(self) -> Dict:
        """
        Detect outliers using ensemble of multiple methods.
        
        Returns:
            Dictionary with combined outlier labels and scores
        """
        print("Detecting outliers using ensemble methods...")
        
        # Run multiple methods
        iso_results = self.detect_outliers_isolation_forest()
        lof_results = self.detect_outliers_lof()
        dbscan_results = self.detect_outliers_dbscan()
        
        # Combine results
        ensemble_labels = (iso_results['labels'] + lof_results['labels'] + dbscan_results['labels']) >= 2
        ensemble_scores = (iso_results['scores'] + lof_results['scores'] + dbscan_results['scores']) / 3
        
        return {
            'labels': ensemble_labels.astype(int),
            'scores': ensemble_scores,
            'method': 'Ensemble',
            'individual_results': {
                'isolation_forest': iso_results,
                'lof': lof_results,
                'dbscan': dbscan_results
            }
        }
    
    def analyze_outliers(self, outlier_results: Dict):
        """Analyze detected outliers and their characteristics."""
        print(f"\nOutlier Analysis - {outlier_results['method']}")
        print("=" * 50)
        
        outlier_labels = outlier_results['labels']
        outlier_scores = outlier_results['scores']
        
        # Basic statistics
        n_outliers = np.sum(outlier_labels)
        n_total = len(outlier_labels)
        outlier_ratio = n_outliers / n_total
        
        print(f"Total repositories: {n_total}")
        print(f"Outliers detected: {n_outliers} ({outlier_ratio:.2%})")
        
        if n_outliers == 0:
            print("No outliers detected!")
            return
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'repo_name': [repo['full_name'] for repo in self.repositories],
            'description': [repo.get('description', '') for repo in self.repositories],
            'stars': [repo['stargazers_count'] for repo in self.repositories],
            'language': [repo.get('language', 'Unknown') for repo in self.repositories],
            'url': [repo['html_url'] for repo in self.repositories],
            'is_outlier': outlier_labels,
            'outlier_score': outlier_scores
        })
        
        # Outlier characteristics
        outliers_df = analysis_df[analysis_df['is_outlier'] == 1].copy()
        outliers_df = outliers_df.sort_values('outlier_score', ascending=True)
        
        print(f"\nTop 10 Most Extreme Outliers:")
        print("-" * 40)
        for i, (_, row) in enumerate(outliers_df.head(10).iterrows()):
            print(f"{i+1}. {row['repo_name']} (Score: {row['outlier_score']:.3f})")
            print(f"   Language: {row['language']}, Stars: {row['stars']}")
            print(f"   Description: {row['description'][:100]}...")
            print()
        
        # Feature analysis for outliers
        outlier_indices = np.where(outlier_labels == 1)[0]
        outlier_features = self.readme_features.iloc[outlier_indices]
        normal_features = self.readme_features.iloc[np.where(outlier_labels == 0)[0]]
        
        # Find features that differ most between outliers and normal
        feature_differences = {}
        for col in self.readme_features.columns:
            if outlier_features[col].std() > 0 and normal_features[col].std() > 0:
                outlier_mean = outlier_features[col].mean()
                normal_mean = normal_features[col].mean()
                
                if normal_mean != 0:
                    relative_diff = abs(outlier_mean - normal_mean) / abs(normal_mean)
                    feature_differences[col] = {
                        'outlier_mean': outlier_mean,
                        'normal_mean': normal_mean,
                        'relative_diff': relative_diff
                    }
        
        # Sort by relative difference
        sorted_features = sorted(feature_differences.items(), 
                               key=lambda x: x[1]['relative_diff'], reverse=True)
        
        print("\nTop 15 Distinguishing Features:")
        print("-" * 40)
        for i, (feature, stats) in enumerate(sorted_features[:15]):
            print(f"{i+1}. {feature}")
            print(f"   Outliers: {stats['outlier_mean']:.3f}, Normal: {stats['normal_mean']:.3f}")
            print(f"   Relative difference: {stats['relative_diff']:.3f}")
            print()
        
        return analysis_df
    
    def visualize_outliers(self, outlier_results: Dict):
        """Visualize outlier detection results."""
        outlier_labels = outlier_results['labels']
        outlier_scores = outlier_results['scores']
        
        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(StandardScaler().fit_transform(self.readme_features))
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: PCA scatter plot
        plt.subplot(2, 3, 1)
        colors = ['blue' if label == 0 else 'red' for label in outlier_labels]
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Outliers in PCA Space')
        plt.legend(['Normal', 'Outlier'])
        
        # Plot 2: Outlier scores distribution
        plt.subplot(2, 3, 2)
        plt.hist(outlier_scores[outlier_labels == 0], bins=30, alpha=0.7, label='Normal', color='blue')
        plt.hist(outlier_scores[outlier_labels == 1], bins=30, alpha=0.7, label='Outlier', color='red')
        plt.xlabel('Outlier Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Outlier Scores')
        plt.legend()
        
        # Plot 3: Feature comparison
        plt.subplot(2, 3, 3)
        outlier_indices = np.where(outlier_labels == 1)[0]
        normal_indices = np.where(outlier_labels == 0)[0]
        
        feature_cols = ['length', 'word_count', 'code_block_count', 'link_count', 'header_count']
        feature_cols = [col for col in feature_cols if col in self.readme_features.columns]
        
        if feature_cols:
            outlier_means = self.readme_features.iloc[outlier_indices][feature_cols].mean()
            normal_means = self.readme_features.iloc[normal_indices][feature_cols].mean()
            
            x = np.arange(len(feature_cols))
            width = 0.35
            
            plt.bar(x - width/2, normal_means, width, label='Normal', color='blue', alpha=0.7)
            plt.bar(x + width/2, outlier_means, width, label='Outlier', color='red', alpha=0.7)
            
            plt.xlabel('Features')
            plt.ylabel('Average Value')
            plt.title('Feature Comparison')
            plt.xticks(x, [col.replace('_', ' ').title() for col in feature_cols], rotation=45)
            plt.legend()
        
        # Plot 4: Repository characteristics
        plt.subplot(2, 3, 4)
        stars = [repo['stargazers_count'] for repo in self.repositories]
        colors = ['blue' if label == 0 else 'red' for label in outlier_labels]
        plt.scatter(range(len(stars)), stars, c=colors, alpha=0.6)
        plt.xlabel('Repository Index')
        plt.ylabel('Stars')
        plt.title('Repository Popularity vs Outlier Status')
        plt.yscale('log')
        
        # Plot 5: Language distribution
        plt.subplot(2, 3, 5)
        languages = [repo.get('language', 'Unknown') for repo in self.repositories]
        lang_outlier_counts = {}
        lang_total_counts = {}
        
        for lang, is_outlier in zip(languages, outlier_labels):
            if lang not in lang_total_counts:
                lang_total_counts[lang] = 0
                lang_outlier_counts[lang] = 0
            lang_total_counts[lang] += 1
            if is_outlier:
                lang_outlier_counts[lang] += 1
        
        # Get top languages
        top_langs = sorted(lang_total_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        lang_names = [lang for lang, _ in top_langs]
        outlier_ratios = [lang_outlier_counts[lang] / lang_total_counts[lang] for lang in lang_names]
        
        plt.bar(lang_names, outlier_ratios, color='red', alpha=0.7)
        plt.xlabel('Programming Language')
        plt.ylabel('Outlier Ratio')
        plt.title('Outlier Ratio by Language')
        plt.xticks(rotation=45)
        
        # Plot 6: Outlier score vs stars
        plt.subplot(2, 3, 6)
        plt.scatter(stars, outlier_scores, c=colors, alpha=0.6)
        plt.xlabel('Stars (log scale)')
        plt.ylabel('Outlier Score')
        plt.title('Outlier Score vs Repository Popularity')
        plt.xscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def generate_outlier_report(self, outlier_results: Dict, filename: str = 'outlier_report.html'):
        """Generate comprehensive HTML report of outlier analysis."""
        print(f"Generating outlier report: {filename}")
        
        outlier_labels = outlier_results['labels']
        outlier_scores = outlier_results['scores']
        
        # Create detailed analysis
        analysis_df = pd.DataFrame({
            'repo_name': [repo['full_name'] for repo in self.repositories],
            'description': [repo.get('description', '') for repo in self.repositories],
            'stars': [repo['stargazers_count'] for repo in self.repositories],
            'language': [repo.get('language', 'Unknown') for repo in self.repositories],
            'url': [repo