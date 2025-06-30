import re
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import markdown
from bs4 import BeautifulSoup
import unicodedata

class ReadmeTextCleaner:
    """
    A comprehensive text cleaner specifically designed for GitHub README.md files.
    Handles markdown formatting, code blocks, links, badges, and other README-specific content.
    """
    
    def __init__(self):
        # Common badges and shields patterns
        self.badge_patterns = [
            r'!\[.*?\]\(https://img\.shields\.io/.*?\)',
            r'!\[.*?\]\(https://badge\.fury\.io/.*?\)',
            r'!\[.*?\]\(https://travis-ci\..*?\)',
            r'!\[.*?\]\(https://circleci\.com/.*?\)',
            r'!\[.*?\]\(https://codecov\.io/.*?\)',
            r'!\[.*?\]\(https://github\.com/.*/workflows/.*/badge\.svg.*?\)',
        ]
        
        # Common README section headers
        self.section_headers = [
            'installation', 'usage', 'examples', 'documentation', 'contributing',
            'license', 'changelog', 'roadmap', 'features', 'requirements',
            'getting started', 'quick start', 'api', 'configuration', 'testing',
            'deployment', 'troubleshooting', 'faq', 'support', 'credits',
            'acknowledgments', 'authors', 'maintainers'
        ]
    
    def clean_readme(self, content: str, options: Dict[str, bool] = None) -> Dict[str, str]:
        """
        Main cleaning function that applies various cleaning operations.
        
        Args:
            content: Raw README content
            options: Dictionary of cleaning options
            
        Returns:
            Dictionary with original and cleaned content plus metadata
        """
        if options is None:
            options = {
                'remove_badges': True,
                'remove_code_blocks': False,
                'remove_links': False,
                'remove_images': True,
                'remove_html': True,
                'normalize_whitespace': True,
                'remove_tables': False,
                'extract_sections': True,
                'preserve_headers': True
            }
        
        cleaned_content = content
        metadata = {}
        
        # Extract metadata before cleaning
        metadata['original_length'] = len(content)
        metadata['line_count'] = len(content.split('\n'))
        
        # Apply cleaning operations
        if options.get('remove_badges', True):
            cleaned_content, badge_count = self._remove_badges(cleaned_content)
            metadata['badges_removed'] = badge_count
        
        if options.get('remove_html', True):
            cleaned_content = self._remove_html_tags(cleaned_content)
        
        if options.get('remove_images', True):
            cleaned_content, img_count = self._remove_images(cleaned_content)
            metadata['images_removed'] = img_count
        
        if options.get('remove_links', False):
            cleaned_content, link_count = self._remove_links(cleaned_content)
            metadata['links_removed'] = link_count
        else:
            cleaned_content = self._clean_links(cleaned_content)
        
        if options.get('remove_code_blocks', False):
            cleaned_content, code_blocks = self._remove_code_blocks(cleaned_content)
            metadata['code_blocks_removed'] = len(code_blocks)
        else:
            cleaned_content = self._clean_code_blocks(cleaned_content)
        
        if options.get('remove_tables', False):
            cleaned_content = self._remove_tables(cleaned_content)
        
        if options.get('normalize_whitespace', True):
            cleaned_content = self._normalize_whitespace(cleaned_content)
        
        # Extract sections if requested
        if options.get('extract_sections', True):
            sections = self._extract_sections(cleaned_content)
            metadata['sections'] = sections
        
        # Clean markdown formatting
        cleaned_content = self._clean_markdown_formatting(cleaned_content)
        
        # Unicode normalization
        cleaned_content = self._normalize_unicode(cleaned_content)
        
        # Final cleanup
        cleaned_content = self._final_cleanup(cleaned_content)
        
        metadata['cleaned_length'] = len(cleaned_content)
        metadata['compression_ratio'] = metadata['cleaned_length'] / metadata['original_length']
        
        return {
            'original': content,
            'cleaned': cleaned_content,
            'metadata': metadata
        }
    
    def _remove_badges(self, content: str) -> Tuple[str, int]:
        """Remove badge/shield images from README"""
        badge_count = 0
        for pattern in self.badge_patterns:
            matches = re.findall(pattern, content)
            badge_count += len(matches)
            content = re.sub(pattern, '', content)
        
        # Generic badge removal
        generic_badge_pattern = r'!\[.*?\]\(https://.*?badge.*?\)'
        matches = re.findall(generic_badge_pattern, content, re.IGNORECASE)
        badge_count += len(matches)
        content = re.sub(generic_badge_pattern, '', content, flags=re.IGNORECASE)
        
        return content, badge_count
    
    def _remove_html_tags(self, content: str) -> str:
        """Remove HTML tags while preserving content"""
        # Handle common HTML elements in README
        html_patterns = [
            r'<details>.*?</details>',  # Collapsible sections
            r'<summary>.*?</summary>',
            r'<br\s*/?>', 
            r'<hr\s*/?>', 
            r'<p>.*?</p>',
            r'<div.*?>.*?</div>',
            r'<span.*?>.*?</span>',
            r'<center>.*?</center>',
            r'<align.*?>.*?</align>'
        ]
        
        for pattern in html_patterns:
            # Extract text content from tags
            content = re.sub(pattern, lambda m: BeautifulSoup(m.group(0), 'html.parser').get_text(), 
                           content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove remaining HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        return content
    
    def _remove_images(self, content: str) -> Tuple[str, int]:
        """Remove image references"""
        # Markdown images
        img_pattern = r'!\[.*?\]\([^)]+\)'
        images = re.findall(img_pattern, content)
        content = re.sub(img_pattern, '', content)
        
        # HTML img tags
        html_img_pattern = r'<img[^>]*>'
        html_images = re.findall(html_img_pattern, content, re.IGNORECASE)
        content = re.sub(html_img_pattern, '', content, flags=re.IGNORECASE)
        
        return content, len(images) + len(html_images)
    
    def _remove_links(self, content: str) -> Tuple[str, int]:
        """Remove all links, keeping only the text"""
        # Markdown links [text](url)
        link_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links = re.findall(link_pattern, content)
        content = re.sub(link_pattern, r'\1', content)
        
        # Plain URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        content = re.sub(url_pattern, '', content)
        
        return content, len(links) + len(urls)
    
    def _clean_links(self, content: str) -> str:
        """Clean links but preserve them"""
        # Replace markdown links with just the text for analysis
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        return content
    
    def _remove_code_blocks(self, content: str) -> Tuple[str, List[str]]:
        """Remove code blocks and return them separately"""
        code_blocks = []
        
        # Fenced code blocks
        fenced_pattern = r'```[\s\S]*?```'
        code_blocks.extend(re.findall(fenced_pattern, content))
        content = re.sub(fenced_pattern, '', content)
        
        # Indented code blocks (4+ spaces)
        indented_pattern = r'\n(?: {4,}|\t).+(?:\n(?: {4,}|\t).+)*'
        code_blocks.extend(re.findall(indented_pattern, content))
        content = re.sub(indented_pattern, '', content)
        
        return content, code_blocks
    
    def _clean_code_blocks(self, content: str) -> str:
        """Replace code blocks with placeholder"""
        # Replace fenced code blocks
        content = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', content)
        
        # Replace inline code
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content
    
    def _remove_tables(self, content: str) -> str:
        """Remove markdown tables"""
        # Simple table detection and removal
        lines = content.split('\n')
        cleaned_lines = []
        in_table = False
        
        for line in lines:
            # Detect table separator line
            if re.match(r'^[\s]*\|?[\s]*:?-+:?[\s]*\|', line):
                in_table = True
                continue
            
            # Check if line looks like table row
            if in_table and '|' in line and line.count('|') >= 2:
                continue
            elif in_table and '|' not in line:
                in_table = False
            
            if not in_table:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract different sections from README"""
        sections = {}
        lines = content.split('\n')
        current_section = 'introduction'
        current_content = []
        
        for line in lines:
            # Check if line is a header
            if re.match(r'^#+\s+', line):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                header_text = re.sub(r'^#+\s+', '', line).lower().strip()
                current_section = self._normalize_section_name(header_text)
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _normalize_section_name(self, header: str) -> str:
        """Normalize section names to standard format"""
        header = header.lower().strip()
        
        # Map common variations to standard names
        section_mapping = {
            'install': 'installation',
            'how to install': 'installation',
            'setup': 'installation',
            'get started': 'getting_started',
            'getting started': 'getting_started',
            'quick start': 'quick_start',
            'quickstart': 'quick_start',
            'how to use': 'usage',
            'user guide': 'usage',
            'example': 'examples',
            'demo': 'examples',
            'contribute': 'contributing',
            'contribution': 'contributing',
            'development': 'contributing',
            'api reference': 'api',
            'api docs': 'api',
            'config': 'configuration',
            'settings': 'configuration',
        }
        
        return section_mapping.get(header, header.replace(' ', '_'))
    
    def _clean_markdown_formatting(self, content: str) -> str:
        """Remove markdown formatting characters"""
        # Headers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        # Bold and italic
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*([^*]+)\*', r'\1', content)      # Italic
        content = re.sub(r'__([^_]+)__', r'\1', content)      # Bold
        content = re.sub(r'_([^_]+)_', r'\1', content)        # Italic
        
        # Lists
        content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # Blockquotes
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
        
        # Horizontal rules
        content = re.sub(r'^[-*_]{3,}$', '', content, flags=re.MULTILINE)
        
        return content
    
    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace and line breaks"""
        # Replace multiple spaces with single space
        content = re.sub(r' +', ' ', content)
        
        # Replace multiple line breaks with maximum of 2
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from lines
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _normalize_unicode(self, content: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode
        content = unicodedata.normalize('NFKD', content)
        
        # Replace common unicode characters
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '…': '...',          # Ellipsis
            '–': '-', '—': '-',  # Dashes
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        return content
    
    def _final_cleanup(self, content: str) -> str:
        """Final cleanup operations"""
        # Remove empty lines at start and end
        content = content.strip()
        
        # Remove lines with only special characters
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines with only punctuation/symbols
            if re.match(r'^[^\w\s]*$', line.strip()) and len(line.strip()) < 5:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_multiple_files(self, file_paths: List[str], options: Dict[str, bool] = None) -> pd.DataFrame:
        """Process multiple README files and return results as DataFrame"""
        results = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                result = self.clean_readme(content, options)
                result['file_path'] = file_path
                result['file_name'] = os.path.basename(file_path)
                
                # Flatten for DataFrame
                flat_result = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'original_content': result['original'],
                    'cleaned_content': result['cleaned'],
                    **result['metadata']
                }
                
                results.append(flat_result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def get_cleaning_stats(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the cleaning process"""
        stats = {
            'total_files': len(df),
            'avg_original_length': df['original_length'].mean(),
            'avg_cleaned_length': df['cleaned_length'].mean(),
            'avg_compression_ratio': df['compression_ratio'].mean(),
            'total_badges_removed': df['badges_removed'].sum() if 'badges_removed' in df.columns else 0,
            'total_images_removed': df['images_removed'].sum() if 'images_removed' in df.columns else 0,
            'files_with_sections': df['sections'].apply(lambda x: len(x) if isinstance(x, dict) else 0).sum()
        }
        
        return stats

# Example usage and utility functions
def example_usage():
    """Example of how to use the ReadmeTextCleaner"""
    
    # Initialize cleaner
    cleaner = ReadmeTextCleaner()
    
    # Example README content
    sample_readme = """
    # My Awesome Project
    
    ![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
    ![License](https://img.shields.io/badge/license-MIT-blue)
    
    This is an **awesome** project that does _amazing_ things.
    
    ## Installation
    
    ```bash
    pip install awesome-project
    ```
    
    ## Usage
    
    Here's how to use it:
    
    ```python
    from awesome_project import AwesomeClass
    obj = AwesomeClass()
    obj.do_something()
    ```
    
    ## Contributing
    
    Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.
    
    <img src="screenshot.png" alt="Screenshot">
    """
    
    # Clean the content
    result = cleaner.clean_readme(sample_readme)
    
    print("Original length:", result['metadata']['original_length'])
    print("Cleaned length:", result['metadata']['cleaned_length'])
    print("Badges removed:", result['metadata']['badges_removed'])
    print("Images removed:", result['metadata']['images_removed'])
    print("\nCleaned content:")
    print(result['cleaned'])
    print("\nSections found:")
    for section, content in result['metadata']['sections'].items():
        print(f"- {section}: {len(content)} characters")

if __name__ == "__main__":
    example_usage()