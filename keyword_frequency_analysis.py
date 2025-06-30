import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class ReadmeKeywordAnalyzer:
    """
    Advanced keyword frequency analysis for GitHub README files.
    Provides multiple analysis methods including TF-IDF, n-grams, and domain-specific analysis.
    """
    
    def __init__(self):
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Tech-specific stopwords
        self.tech_stopwords = {
            'use', 'using', 'used', 'make', 'makes', 'made', 'get', 'getting', 'got',
            'run', 'running', 'ran', 'work', 'working', 'worked', 'need', 'needs',
            'want', 'wants', 'like', 'one', 'two', 'first', 'last', 'new', 'old',
            'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'way',
            'ways', 'time', 'times', 'thing', 'things', 'part', 'parts', 'place',
            'places', 'right', 'left', 'back', 'front', 'start', 'end', 'begin',
            'finish', 'done', 'doing', 'go', 'going', 'come', 'coming', 'take',
            'taking', 'give', 'giving', 'put', 'putting', 'set', 'setting'
        }
        
        # Programming language keywords
        self.programming_languages = {
            'python', 'javascript', 'java', 'cpp', 'csharp', 'php', 'ruby', 'go',
            'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'lua',
            'typescript', 'dart', 'elixir', 'haskell', 'clojure', 'erlang'
        }
        
        # Framework and library patterns
        self.tech_patterns = {
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'express', 'spring',
                'rails', 'laravel', 'symfony', 'bootstrap', 'jquery', 'nodejs',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
                'cassandra', 'elasticsearch', 'firebase', 'dynamodb'
            ],
            'tools': [
                'docker', 'kubernetes', 'jenkins', 'travis', 'circleci', 'webpack',
                'babel', 'eslint', 'jest', 'mocha', 'cypress', 'selenium'
            ],
            'platforms': [
                'aws', 'azure', 'gcp', 'heroku', 'netlify', 'vercel', 'github',
                'gitlab', 'bitbucket', 'linux', 'windows', 'macos', 'android', 'ios'
            ]
        }
        
        # Common README sections for context analysis
        self.readme_sections = [
            'installation', 'usage', 'documentation', 'contributing', 'license',
            'features', 'requirements', 'configuration', 'api', 'examples'
        ]
    
    def analyze_keywords(self, 
                        texts: Union[str, List[str]], 
                        analysis_type: str = 'comprehensive',
                        min_freq: int = 2,
                        max_features: int = 1000,
                        ngram_range: Tuple[int, int] = (1, 3)) -> Dict:
        """
        Main keyword analysis function with multiple analysis types.
        
        Args:
            texts: Single text or list of texts to analyze
            analysis_type: 'basic', 'advanced', 'comprehensive', 'tech_focused'
            min_freq: Minimum frequency for keywords
            max_features: Maximum number of features to extract
            ngram_range: Range for n-gram analysis
            
        Returns:
            Dictionary containing various analysis results
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'analysis_type': analysis_type,
            'document_count': len(texts),
            'total_words': sum(len(text.split()) for text in texts)
        }
        
        # Basic frequency analysis
        if analysis_type in ['basic', 'advanced', 'comprehensive']:
            results.update(self._basic_frequency_analysis(texts, min_freq))
        
        # Advanced analysis
        if analysis_type in ['advanced', 'comprehensive']:
            results.update(self._advanced_analysis(texts, max_features, ngram_range))
        
        # Comprehensive analysis
        if analysis_type == 'comprehensive':
            results.update(self._comprehensive_analysis(texts))
        
        # Tech-focused analysis
        if analysis_type == 'tech_focused':
            results.update(self._tech_focused_analysis(texts))
        
        return results
    
    def _basic_frequency_analysis(self, texts: List[str], min_freq: int) -> Dict:
        """Basic word frequency analysis"""
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Clean and tokenize
        cleaned_tokens = self._clean_and_tokenize(combined_text)
        
        # Word frequency
        word_freq = Counter(cleaned_tokens)
        filtered_freq = {word: count for word, count in word_freq.items() 
                        if count >= min_freq}
        
        # Character frequency
        char_freq = Counter(char.lower() for char in combined_text if char.isalpha())
        
        return {
            'word_frequencies': dict(word_freq.most_common(50)),
            'filtered_frequencies': filtered_freq,
            'character_frequencies': dict(char_freq.most_common(26)),
            'unique_words': len(word_freq),
            'vocabulary_richness': len(word_freq) / len(cleaned_tokens) if cleaned_tokens else 0
        }
    
    def _advanced_analysis(self, texts: List[str], max_features: int, ngram_range: Tuple[int, int]) -> Dict:
        """Advanced analysis with TF-IDF and n-grams"""
        results = {}
        
        # TF-IDF Analysis
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            tfidf_scores = list(zip(feature_names, mean_scores))
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            
            results['tfidf_keywords'] = tfidf_scores[:50]
            results['tfidf_matrix_shape'] = tfidf_matrix.shape
        except Exception as e:
            results['tfidf_error'] = str(e)
        
        # N-gram analysis
        results['ngrams'] = self._extract_ngrams(texts, ngram_range)
        
        # POS tagging analysis
        results['pos_analysis'] = self._pos_analysis(texts)
        
        return results
    
    def _comprehensive_analysis(self, texts: List[str]) -> Dict:
        """Comprehensive analysis including domain-specific insights"""
        results = {}
        
        # Technology stack analysis
        results['tech_stack'] = self._analyze_tech_stack(texts)
        
        # Section-based analysis
        results['section_analysis'] = self._analyze_by_sections(texts)
        
        # Sentiment and complexity analysis
        results['text_metrics'] = self._calculate_text_metrics(texts)
        
        # Topic modeling (simplified)
        results['topics'] = self._simple_topic_analysis(texts)
        
        # Named entity recognition
        results['entities'] = self._extract_entities(texts)
        
        return results
    
    def _tech_focused_analysis(self, texts: List[str]) -> Dict:
        """Technology-focused keyword analysis"""
        results = {}
        
        # Programming language detection
        results['programming_languages'] = self._detect_programming_languages(texts)
        
        # Framework and tool analysis
        results['frameworks_tools'] = self._analyze_frameworks_tools(texts)
        
        # Technical term frequency
        results['technical_terms'] = self._extract_technical_terms(texts)
        
        # Version and dependency analysis
        results['versions_dependencies'] = self._analyze_versions_dependencies(texts)
        
        return results
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """Clean text and return tokens"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english')).union(self.tech_stopwords)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return tokens
    
    def _extract_ngrams(self, texts: List[str], ngram_range: Tuple[int, int]) -> Dict:
        """Extract n-grams from texts"""
        ngram_results = {}
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                stop_words='english',
                max_features=100,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            try:
                ngram_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                frequencies = np.sum(ngram_matrix.toarray(), axis=0)
                
                ngram_freq = list(zip(feature_names, frequencies))
                ngram_freq.sort(key=lambda x: x[1], reverse=True)
                
                ngram_results[f'{n}_grams'] = ngram_freq[:20]
            except Exception as e:
                ngram_results[f'{n}_grams_error'] = str(e)
        
        return ngram_results
    
    def _pos_analysis(self, texts: List[str]) -> Dict:
        """Part-of-speech analysis"""
        pos_counts = Counter()
        important_words = {'nouns': [], 'adjectives': [], 'verbs': []}
        
        for text in texts[:5]:  # Limit for performance
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            for word, pos in pos_tags:
                if len(word) > 2 and word.isalpha():
                    pos_counts[pos] += 1
                    
                    if pos.startswith('NN'):  # Nouns
                        important_words['nouns'].append(word)
                    elif pos.startswith('JJ'):  # Adjectives
                        important_words['adjectives'].append(word)
                    elif pos.startswith('VB'):  # Verbs
                        important_words['verbs'].append(word)
        
        # Get most common words by POS
        for pos_type in important_words:
            word_counts = Counter(important_words[pos_type])
            important_words[pos_type] = word_counts.most_common(10)
        
        return {
            'pos_distribution': dict(pos_counts.most_common(20)),
            'important_words_by_pos': important_words
        }
    
    def _analyze_tech_stack(self, texts: List[str]) -> Dict:
        """Analyze technology stack mentioned in READMEs"""
        combined_text = ' '.join(texts).lower()
        tech_stack = {}
        
        for category, technologies in self.tech_patterns.items():
            found_tech = []
            for tech in technologies:
                # Count occurrences with word boundaries
                pattern = r'\b' + re.escape(tech) + r'\b'
                matches = len(re.findall(pattern, combined_text))
                if matches > 0:
                    found_tech.append((tech, matches))
            
            tech_stack[category] = sorted(found_tech, key=lambda x: x[1], reverse=True)
        
        return tech_stack
    
    def _analyze_by_sections(self, texts: List[str]) -> Dict:
        """Analyze keywords by README sections"""
        section_analysis = {}
        
        for text in texts:
            sections = self._extract_text_sections(text)
            for section_name, section_content in sections.items():
                if section_name not in section_analysis:
                    section_analysis[section_name] = []
                
                tokens = self._clean_and_tokenize(section_content)
                word_freq = Counter(tokens)
                section_analysis[section_name].extend(tokens)
        
        # Get top keywords per section
        for section in section_analysis:
            word_freq = Counter(section_analysis[section])
            section_analysis[section] = word_freq.most_common(10)
        
        return section_analysis
    
    def _extract_text_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text based on headers"""
        sections = {}
        lines = text.split('\n')
        current_section = 'introduction'
        current_content = []
        
        for line in lines:
            if re.match(r'^#+\s+', line):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                header = re.sub(r'^#+\s+', '', line).lower().strip()
                current_section = header.replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _calculate_text_metrics(self, texts: List[str]) -> Dict:
        """Calculate various text complexity metrics"""
        metrics = {
            'avg_sentence_length': [],
            'avg_word_length': [],
            'readability_score': [],
            'technical_density': []
        }
        
        for text in texts:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            
            # Average sentence length
            if sentences:
                avg_sent_len = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences)
                metrics['avg_sentence_length'].append(avg_sent_len)
            
            # Average word length
            if words:
                avg_word_len = sum(len(word) for word in words if word.isalpha()) / len([w for w in words if w.isalpha()])
                metrics['avg_word_length'].append(avg_word_len)
            
            # Technical density (technical terms / total words)
            tech_terms = self._count_technical_terms(text)
            if words:
                tech_density = tech_terms / len(words)
                metrics['technical_density'].append(tech_density)
        
        # Calculate averages
        for metric in metrics:
            if metrics[metric]:
                metrics[metric] = {
                    'average': np.mean(metrics[metric]),
                    'std': np.std(metrics[metric]),
                    'min': np.min(metrics[metric]),
                    'max': np.max(metrics[metric])
                }
        
        return metrics
    
    def _simple_topic_analysis(self, texts: List[str]) -> Dict:
        """Simple topic analysis using keyword clustering"""
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Simple clustering based on co-occurrence
            topics = {}
            
            # Get top terms for each document
            for i, doc_vector in enumerate(tfidf_matrix.toarray()):
                top_indices = doc_vector.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices if doc_vector[idx] > 0]
                topics[f'document_{i}'] = top_terms
            
            return topics
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_entities(self, texts: List[str]) -> Dict:
        """Extract named entities from texts"""
        entities = {'PERSON': [], 'ORGANIZATION': [], 'GPE': [], 'TECHNOLOGY': []}
        
        for text in texts[:3]:  # Limit for performance
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            try:
                chunks = ne_chunk(pos_tags)
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_name = ' '.join([token for token, pos in chunk.leaves()])
                        entity_type = chunk.label()
                        if entity_type in entities:
                            entities[entity_type].append(entity_name)
            except:
                continue
        
        # Count occurrences
        for entity_type in entities:
            entity_counts = Counter(entities[entity_type])
            entities[entity_type] = entity_counts.most_common(10)
        
        return entities
    
    def _detect_programming_languages(self, texts: List[str]) -> Dict:
        """Detect programming languages mentioned"""
        combined_text = ' '.join(texts).lower()
        detected_languages = {}
        
        for lang in self.programming_languages:
            pattern = r'\b' + re.escape(lang) + r'\b'
            count = len(re.findall(pattern, combined_text))
            if count > 0:
                detected_languages[lang] = count
        
        # Sort by frequency
        sorted_languages = sorted(detected_languages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'detected_languages': sorted_languages,
            'primary_language': sorted_languages[0][0] if sorted_languages else None,
            'language_diversity': len(detected_languages)
        }
    
    def _analyze_frameworks_tools(self, texts: List[str]) -> Dict:
        """Analyze frameworks and tools mentioned"""
        combined_text = ' '.join(texts).lower()
        results = {}
        
        all_tools = []
        for category, tools in self.tech_patterns.items():
            all_tools.extend(tools)
        
        tool_counts = {}
        for tool in all_tools:
            pattern = r'\b' + re.escape(tool) + r'\b'
            count = len(re.findall(pattern, combined_text))
            if count > 0:
                tool_counts[tool] = count
        
        results['tool_frequencies'] = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        results['total_unique_tools'] = len(tool_counts)
        
        return results
    
    def _extract_technical_terms(self, texts: List[str]) -> Dict:
        """Extract technical terms using patterns"""
        combined_text = ' '.join(texts).lower()
        
        # Patterns for technical terms
        patterns = {
            'version_numbers': r'\b\d+\.\d+(?:\.\d+)?\b',
            'file_extensions': r'\.\w{2,4}\b',
            'urls': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'email_patterns': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'technical_acronyms': r'\b[A-Z]{2,6}\b',
            'camel_case': r'\b[a-z]+(?:[A-Z][a-z]*)+\b',
            'snake_case': r'\b[a-z]+(?:_[a-z]+)+\b'
        }
        
        technical_terms = {}
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, combined_text)
            if matches:
                technical_terms[pattern_name] = Counter(matches).most_common(10)
        
        return technical_terms
    
    def _analyze_versions_dependencies(self, texts: List[str]) -> Dict:
        """Analyze version numbers and dependencies"""
        combined_text = ' '.join(texts)
        
        # Version patterns
        version_pattern = r'(?:version|v\.?|ver\.?)\s*:?\s*(\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?)'
        versions = re.findall(version_pattern, combined_text, re.IGNORECASE)
        
        # Dependency patterns (common package managers)
        dependency_patterns = {
            'npm': r'npm install\s+([\w\-@/]+)',
            'pip': r'pip install\s+([\w\-\[\]]+)',
            'composer': r'composer require\s+([\w\-/]+)',
            'gem': r'gem install\s+([\w\-]+)',
            'go_get': r'go get\s+([\w\-./]+)'
        }
        
        dependencies = {}
        for manager, pattern in dependency_patterns.items():
            deps = re.findall(pattern, combined_text, re.IGNORECASE)
            if deps:
                dependencies[manager] = Counter(deps).most_common(10)
        
        return {
            'versions_found': Counter(versions).most_common(10),
            'dependencies_by_manager': dependencies
        }
    
    def _count_technical_terms(self, text: str) -> int:
        """Count technical terms in text"""
        technical_indicators = [
            r'\b(?:api|sdk|cli|gui|ui|ux|db|sql|json|xml|html|css|js)\b',
            r'\b(?:server|client|backend|frontend|database|framework)\b',
            r'\b(?:algorithm|function|method|class|object|variable)\b',
            r'\b(?:deploy|build|compile|debug|test|optimize)\b'
        ]
        
        count = 0
        text_lower = text.lower()
        for pattern in technical_indicators:
            count += len(re.findall(pattern, text_lower))
        
        return count
    
    def generate_visualizations(self, analysis_results: Dict, output_dir: str = None) -> Dict:
        """Generate various visualizations for the keyword analysis"""
        visualizations = {}
        
        # Word frequency bar chart
        if 'word_frequencies' in analysis_results:
            fig = self._create_frequency_chart(analysis_results['word_frequencies'])
            visualizations['word_frequency_chart'] = fig
        
        # Technology stack visualization
        if 'tech_stack' in analysis_results:
            fig = self._create_tech_stack_chart(analysis_results['tech_stack'])
            visualizations['tech_stack_chart'] = fig
        
        # Word cloud
        if 'word_frequencies' in analysis_results:
            wordcloud = self._create_wordcloud(analysis_results['word_frequencies'])
            visualizations['wordcloud'] = wordcloud
        
        # N-grams visualization
        if 'ngrams' in analysis_results:
            fig = self._create_ngrams_chart(analysis_results['ngrams'])
            visualizations['ngrams_chart'] = fig
        
        return visualizations
    
    def _create_frequency_chart(self, word_frequencies: Dict) -> go.Figure:
        """Create interactive frequency chart"""
        words = list(word_frequencies.keys())[:20]
        frequencies = list(word_frequencies.values())[:20]
        
        fig = go.Figure(data=[
            go.Bar(x=frequencies, y=words, orientation='h',
                   marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Top 20 Word Frequencies',
            xaxis_title='Frequency',
            yaxis_title='Words',
            height=600
        )
        
        return fig
    
    def _create_tech_stack_chart(self, tech_stack: Dict) -> go.Figure:
        """Create technology stack visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(tech_stack.keys())[:4],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (category, technologies) in enumerate(list(tech_stack.items())[:4]):
            if technologies:
                techs = [tech[0] for tech in technologies[:10]]
                counts = [tech[1] for tech in technologies[:10]]
                
                row, col = positions[i]
                fig.add_trace(
                    go.Bar(x=techs, y=counts, name=category),
                    row=row, col=col
                )
        
        fig.update_layout(height=800, showlegend=False)
        return fig
    
    def _create_wordcloud(self, word_frequencies: Dict) -> WordCloud:
        """Create word cloud visualization"""
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_frequencies)
        
        return wordcloud
    
    def _create_ngrams_chart(self, ngrams: Dict) -> go.Figure:
        """Create n-grams visualization"""
        fig = make_subplots(
            rows=len(ngrams), cols=1,
            subplot_titles=list(ngrams.keys())
        )
        
        for i, (ngram_type, ngram_data) in enumerate(ngrams.items()):
            if isinstance(ngram_data, list) and ngram_data:
                phrases = [item[0] for item in ngram_data[:10]]
                frequencies = [item[1] for item in ngram_data[:10]]
                
                fig.add_trace(
                    go.Bar(x=frequencies, y=phrases, orientation='h'),
                    row=i+1, col=1
                )
        
        fig.update_layout(height=300*len(ngrams), showlegend=False)
        return fig

# Example usage and utility functions
def example_usage():
    """Example of how to use the ReadmeKeywordAnalyzer"""
    
    # Initialize analyzer
    analyzer = ReadmeKeywordAnalyzer()
    
    # Sample README texts
    sample_texts = [
        """
        # Machine Learning Project
        
        This is a Python machine learning project using TensorFlow and scikit-learn.
        
        ## Installation
        
        ```bash
        pip install tensorflow scikit-learn pandas numpy
        ```
        
        ## Usage
        
        The model uses neural networks and supports both classification and regression tasks.
        """,
        """
        # Web Development Framework
        
        A React-based web application with Node.js backend.
        
        ## Features
        
        - RESTful API
        - Real-time updates
        - Docker support
        - MongoDB integration
        
        ## Installation
        
        ```bash
        npm install
        docker-compose up
        ```
        """
    ]
    
    # Comprehensive analysis
    results = analyzer.analyze_keywords(
        sample_texts, 
        analysis_type='comprehensive',
        min_freq=1,
        ngram_range=(1, 3)
    )
    
    # Print results
    print("=== KEYWORD ANALYSIS RESULTS ===")
    print(f"Documents analyzed: {results['document_count']}")
    print(f"Total words: {results['total_words']}")
    print(f"Unique words: {results['unique_words']}")
    print(f"Vocabulary richness: {results['vocabulary_richness']:.3f}")
    
    print("\n=== TOP WORDS ===")
    for word, freq in list(results['word_frequencies'].items())[:10]:
        print(f"{word}: {freq}")
    
    print("\n=== TECHNOLOGY STACK ===")
    if 'tech_stack' in results:
        for category, technologies in results['tech_stack'].items():
            if technologies:
                print(f"\n{category.upper()}:")
                for tech, count in technologies[:5]:
                    print(f"  {tech}: {count}")
    
    print("\n=== PROGRAMMING LANGUAGES ===")
    if 'programming_languages' in results:
        langs = results['programming_languages']['detected_languages']
        for lang, count in langs[:5]:
            print(f"{lang}: {count}")
    
    print("\n=== N-GRAMS ===")
    if 'ngrams' in results:
        for ngram_type, ngrams in results['ngrams'].items():
            if isinstance(ngrams, list) and ngrams:
                print(f"\n{ngram_type.upper()}:")
                for phrase, freq in ngrams[:5]:
                    print(f"  '{phrase}': {freq}")
    
    print("\n=== TF-IDF KEYWORDS ===")
    if 'tfidf_keywords' in results:
        for keyword, score in results['tfidf_keywords'][:10]:
            print(f"{keyword}: {score:.4f}")
    
    print("\n=== SECTION ANALYSIS ===")
    if 'section_analysis' in results:
        for section, keywords in results['section_analysis'].items():
            if keywords:
                print(f"\n{section.upper()}:")
                for word, freq in keywords[:3]:
                    print(f"  {word}: {freq}")

def batch_analysis_example():
    """Example of batch analysis with multiple README files"""
    
    analyzer = ReadmeKeywordAnalyzer()
    
    # Simulate multiple README files
    readme_files = {
        'ml_project': """
        # Deep Learning Image Classification
        
        This project implements a convolutional neural network using TensorFlow and Keras 
        for image classification tasks. The model achieves 95% accuracy on CIFAR-10 dataset.
        
        ## Features
        - Data preprocessing and augmentation
        - Transfer learning with pre-trained models
        - Model evaluation and visualization
        - GPU acceleration support
        
        ## Requirements
        - Python 3.8+
        - TensorFlow 2.x
        - NumPy, Pandas, Matplotlib
        - CUDA for GPU support
        
        ## Installation
        ```bash
        pip install tensorflow pandas numpy matplotlib scikit-learn
        ```
        """,
        
        'web_app': """
        # E-commerce Platform
        
        A full-stack e-commerce application built with React, Node.js, and MongoDB.
        Includes user authentication, payment processing, and admin dashboard.
        
        ## Tech Stack
        - Frontend: React, Redux, Material-UI
        - Backend: Node.js, Express.js
        - Database: MongoDB, Redis
        - Payment: Stripe API
        - Deployment: Docker, AWS
        
        ## Features
        - User registration and authentication
        - Product catalog with search and filtering
        - Shopping cart and checkout process
        - Order management system
        - Admin panel for inventory management
        
        ## Installation
        ```bash
        npm install
        docker-compose up -d
        npm run dev
        ```
        """,
        
        'devops_tool': """
        # Kubernetes Deployment Automation
        
        An automated deployment tool for Kubernetes clusters with CI/CD integration.
        Supports multiple cloud providers and infrastructure as code.
        
        ## Features
        - Automated cluster provisioning
        - GitOps workflow integration
        - Multi-cloud support (AWS, GCP, Azure)
        - Monitoring and alerting
        - Security scanning and compliance
        
        ## Requirements
        - Kubernetes 1.20+
        - Helm 3.x
        - Docker
        - kubectl
        
        ## Supported Platforms
        - Amazon EKS
        - Google GKE
        - Azure AKS
        - On-premises clusters
        
        ## Installation
        ```bash
        curl -fsSL https://get.helm.sh/helm-v3.0.0-linux-amd64.tar.gz | tar -xz
        kubectl apply -f deploy/
        ```
        """
    }
    
    # Analyze all README files
    texts = list(readme_files.values())
    
    # Different types of analysis
    analysis_types = ['basic', 'advanced', 'comprehensive', 'tech_focused']
    
    for analysis_type in analysis_types:
        print(f"\n{'='*50}")
        print(f"ANALYSIS TYPE: {analysis_type.upper()}")
        print(f"{'='*50}")
        
        results = analyzer.analyze_keywords(texts, analysis_type=analysis_type)
        
        # Display key metrics
        print(f"Documents: {results['document_count']}")
        print(f"Total words: {results['total_words']}")
        
        if 'word_frequencies' in results:
            print(f"Unique words: {results['unique_words']}")
            print(f"Vocabulary richness: {results['vocabulary_richness']:.3f}")
            
            print("\nTop 10 Words:")
            for word, freq in list(results['word_frequencies'].items())[:10]:
                print(f"  {word}: {freq}")
        
        if 'tech_stack' in results:
            print("\nTechnology Stack:")
            for category, techs in results['tech_stack'].items():
                if techs:
                    print(f"  {category}: {', '.join([f'{t[0]}({t[1]})' for t in techs[:3]])}")
        
        if 'programming_languages' in results:
            langs = results['programming_languages']['detected_languages']
            if langs:
                print(f"\nPrimary Language: {results['programming_languages']['primary_language']}")
                print(f"Language Diversity: {results['programming_languages']['language_diversity']}")
        
        if 'tfidf_keywords' in results:
            print("\nTF-IDF Top Keywords:")
            for keyword, score in results['tfidf_keywords'][:5]:
                print(f"  {keyword}: {score:.4f}")

def comparative_analysis_example():
    """Example of comparative analysis between different project types"""
    
    analyzer = ReadmeKeywordAnalyzer()
    
    # Different project categories
    project_categories = {
        'machine_learning': [
            "Deep learning framework with TensorFlow and PyTorch support. Includes neural networks, data preprocessing, and model evaluation.",
            "Computer vision project using OpenCV and scikit-learn. Features image classification, object detection, and face recognition.",
            "Natural language processing toolkit with NLTK and spaCy. Supports text classification, sentiment analysis, and named entity recognition."
        ],
        
        'web_development': [
            "React-based web application with Node.js backend. Uses Express.js, MongoDB, and JWT authentication.",
            "Vue.js frontend with Django REST API. Includes user management, real-time chat, and payment integration.",
            "Angular application with Spring Boot backend. Features microservices architecture and MySQL database."
        ],
        
        'devops_infrastructure': [
            "Docker containerization with Kubernetes orchestration. Includes CI/CD pipelines using Jenkins and GitLab.",
            "Terraform infrastructure as code for AWS, Azure, and GCP. Supports auto-scaling and monitoring.",
            "Ansible automation for server configuration. Integrates with Prometheus, Grafana, and ELK stack."
        ]
    }
    
    print("=== COMPARATIVE ANALYSIS BY PROJECT TYPE ===\n")
    
    category_results = {}
    
    for category, texts in project_categories.items():
        print(f"Analyzing {category.replace('_', ' ').title()}...")
        
        results = analyzer.analyze_keywords(
            texts, 
            analysis_type='tech_focused',
            min_freq=1
        )
        
        category_results[category] = results
        
        print(f"  Total words: {results['total_words']}")
        if 'programming_languages' in results:
            primary_lang = results['programming_languages']['primary_language']
            lang_diversity = results['programming_languages']['language_diversity']
            print(f"  Primary language: {primary_lang}")
            print(f"  Language diversity: {lang_diversity}")
        
        if 'frameworks_tools' in results:
            total_tools = results['frameworks_tools']['total_unique_tools']
            print(f"  Unique tools/frameworks: {total_tools}")
        
        print()
    
    # Compare categories
    print("=== CATEGORY COMPARISON ===")
    
    # Language diversity comparison
    print("\nLanguage Diversity by Category:")
    for category, results in category_results.items():
        if 'programming_languages' in results:
            diversity = results['programming_languages']['language_diversity']
            print(f"  {category.replace('_', ' ').title()}: {diversity} languages")
    
    # Most common technologies per category
    print("\nTop Technologies by Category:")
    for category, results in category_results.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        if 'frameworks_tools' in results:
            tools = results['frameworks_tools']['tool_frequencies'][:5]
            for tool, count in tools:
                print(f"  {tool}: {count}")

def export_analysis_results(results: Dict, filename: str = 'keyword_analysis_results.json'):
    """Export analysis results to JSON file"""
    import json
    
    # Convert non-serializable objects to serializable format
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            serializable_results[key] = value
        elif hasattr(value, 'tolist'):  # NumPy arrays
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = str(value)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis results exported to {filename}")

def create_analysis_report(results: Dict, project_name: str = "README Analysis") -> str:
    """Create a comprehensive analysis report"""
    
    report = f"""
# {project_name} - Keyword Analysis Report

## Executive Summary

- **Documents Analyzed**: {results.get('document_count', 'N/A')}
- **Total Words**: {results.get('total_words', 'N/A'):,}
- **Unique Words**: {results.get('unique_words', 'N/A'):,}
- **Vocabulary Richness**: {results.get('vocabulary_richness', 0):.3f}

## Key Findings

### Most Frequent Words
"""
    
    if 'word_frequencies' in results:
        report += "\n| Word | Frequency |\n|------|----------|\n"
        for word, freq in list(results['word_frequencies'].items())[:15]:
            report += f"| {word} | {freq} |\n"
    
    if 'tech_stack' in results:
        report += "\n### Technology Stack Analysis\n"
        for category, technologies in results['tech_stack'].items():
            if technologies:
                report += f"\n#### {category.title()}\n"
                for tech, count in technologies[:10]:
                    report += f"- **{tech}**: {count} mentions\n"
    
    if 'programming_languages' in results:
        report += "\n### Programming Languages\n"
        langs = results['programming_languages']['detected_languages']
        primary = results['programming_languages']['primary_language']
        diversity = results['programming_languages']['language_diversity']
        
        report += f"- **Primary Language**: {primary}\n"
        report += f"- **Language Diversity**: {diversity} different languages\n"
        report += "\n#### Language Distribution\n"
        for lang, count in langs[:10]:
            report += f"- {lang}: {count} mentions\n"
    
    if 'tfidf_keywords' in results:
        report += "\n### TF-IDF Important Keywords\n"
        report += "\n| Keyword | TF-IDF Score |\n|---------|-------------|\n"
        for keyword, score in results['tfidf_keywords'][:15]:
            report += f"| {keyword} | {score:.4f} |\n"
    
    if 'ngrams' in results:
        report += "\n### N-Gram Analysis\n"
        for ngram_type, ngrams in results['ngrams'].items():
            if isinstance(ngrams, list) and ngrams:
                report += f"\n#### {ngram_type.replace('_', '-').title()}\n"
                for phrase, freq in ngrams[:10]:
                    report += f"- '{phrase}': {freq}\n"
    
    if 'section_analysis' in results:
        report += "\n### Section-Based Analysis\n"
        for section, keywords in results['section_analysis'].items():
            if keywords:
                report += f"\n#### {section.replace('_', ' ').title()}\n"
                for word, freq in keywords[:5]:
                    report += f"- {word}: {freq}\n"
    
    if 'text_metrics' in results:
        report += "\n### Text Complexity Metrics\n"
        metrics = results['text_metrics']
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                report += f"- **{metric_name.replace('_', ' ').title()}**: "
                report += f"Avg: {metric_data.get('average', 0):.2f}, "
                report += f"Std: {metric_data.get('std', 0):.2f}\n"
    
    report += f"\n---\n*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    return report

# Main execution
if __name__ == "__main__":
    print("Running README Keyword Analysis Examples...\n")
    
    # Basic example
    print("=== BASIC EXAMPLE ===")
    example_usage()
    
    # Batch analysis
    print("\n=== BATCH ANALYSIS EXAMPLE ===")
    batch_analysis_example()
    
    # Comparative analysis
    print("\n=== COMPARATIVE ANALYSIS EXAMPLE ===")
    comparative_analysis_example()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Use the ReadmeKeywordAnalyzer class to analyze your own README files!")
    print("Key methods:")
    print("- analyze_keywords(): Main analysis function")
    print("- generate_visualizations(): Create charts and plots")
    print("- process_multiple_files(): Batch processing")
    print("- create_analysis_report(): Generate comprehensive reports")