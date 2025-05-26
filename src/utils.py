import re
import sys

class TranscriptLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chunks = []
        self.timestamps = []
    
    def load_and_chunk(self):
        """Load transcript and split into timestamped chunks"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse timestamped transcript using regex
            # Pattern matches [HH:MM - HH:MM] followed by text
            pattern = r'\[(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})\]\s*(.+?)(?=\[|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for timestamp, text in matches:
                # Clean and preprocess text
                cleaned_text = self.clean_text(text)
                if cleaned_text:  # Only add non-empty chunks
                    self.timestamps.append(timestamp)
                    self.chunks.append(cleaned_text)
            
            if not self.chunks:
                raise ValueError("No valid transcript chunks found. Check file format.")
            
            print(f"Loaded {len(self.chunks)} transcript chunks")
            return self.chunks, self.timestamps
            
        except FileNotFoundError:
            print(f"Error: Could not find file {self.file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading transcript: {e}")
            sys.exit(1)
    
    def clean_text(self, text):
        """Clean and preprocess text chunk"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove any remaining unwanted characters
        text = re.sub(r'[^\w\s\.\,\?\!\-\']', '', text)
        
        return text

    def combine_chunks(self, chunk_indices, context_window=1):
        """Combine multiple chunks with optional context"""
        if not chunk_indices:
            return None, None
        
        # Sort indices
        indices = sorted(set(chunk_indices))
        
        # Expand with context window
        expanded_indices = []
        for idx in indices:
            start = max(0, idx - context_window)
            end = min(len(self.chunks), idx + context_window + 1)
            expanded_indices.extend(range(start, end))
        
        # Remove duplicates and sort
        expanded_indices = sorted(set(expanded_indices))
        
        # Combine texts and get timestamp range
        combined_text = ' '.join([self.chunks[i] for i in expanded_indices])
        start_timestamp = self.timestamps[expanded_indices[0]]
        end_timestamp = self.timestamps[expanded_indices[-1]]
        
        # Create combined timestamp
        start_time = start_timestamp.split(' - ')[0]
        end_time = end_timestamp.split(' - ')[1]
        combined_timestamp = f"{start_time} - {end_time}"
        
        return combined_timestamp, combined_text