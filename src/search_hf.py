import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class HuggingFaceSearcher:
    def __init__(self, chunks, timestamps, model_name='all-MiniLM-L6-v2'):
        self.chunks = chunks
        self.timestamps = timestamps
        self.model_name = model_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self._load_model_and_embed()
    
    def _load_model_and_embed(self):
        """Load the sentence transformer model and create embeddings"""
        print(f"Loading HuggingFace model: {self.model_name}")
        
        try:
            # Load pre-trained sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
            
            # Create embeddings for all chunks
            print("Creating embeddings for transcript chunks...")
            self.embeddings = self.model.encode(
                self.chunks,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32
            )
            
            print(f"Created embeddings with shape: {self.embeddings.shape}")
            
        except Exception as e:
            print(f"Error loading model or creating embeddings: {e}")
            raise
    
    def search(self, query, top_k=1, similarity_threshold=0.3):
        """
        Search for the most relevant chunk using semantic embeddings
        
        Args:
            query (str): User question
            top_k (int): Number of top results to consider
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            tuple: (timestamp, text) of most relevant chunk or None
        """
        if not query.strip():
            return None
        
        try:
            # Create embedding for the query
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            embeddings = np.asarray(self.embeddings)
            query_embedding = np.asarray(query_embedding)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
            
            # Get top k most similar documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Check if best match meets threshold
            best_idx = top_indices[0]
            best_similarity = similarities[best_idx]
            
            if best_similarity < similarity_threshold:
                return None
            
            print(f"Best match similarity: {best_similarity:.4f}")
            
            return self.timestamps[best_idx], self.chunks[best_idx]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return None
    
    def search_multiple(self, query, top_k=3, similarity_threshold=0.3):
        """
        Search for multiple relevant chunks
        
        Args:
            query (str): User question
            top_k (int): Number of results to return
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            list: List of (timestamp, text, score) tuples
        """
        if not query.strip():
            return []
        
        try:
            # Create embedding for the query
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            
            # Calculate similarities
            embeddings = np.asarray(self.embeddings)
            query_embedding = np.asarray(query_embedding)
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by threshold and return results
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= similarity_threshold:
                    results.append((
                        self.timestamps[idx],
                        self.chunks[idx],
                        score
                    ))
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def search_with_context(self, query, context_window=1, similarity_threshold=0.3):
        """
        Search and return results with surrounding context
        
        Args:
            query (str): User question
            context_window (int): Number of chunks before/after to include
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            tuple: (timestamp_range, extended_text) or None
        """
        result = self.search(query, similarity_threshold=similarity_threshold)
        if not result:
            return None
        
        timestamp, text = result
        
        # Find the index of the best match
        best_idx = None
        for i, chunk in enumerate(self.chunks):
            if chunk == text:
                best_idx = i
                break
        
        if best_idx is None:
            return result
        
        # Get context window
        start_idx = max(0, best_idx - context_window)
        end_idx = min(len(self.chunks), best_idx + context_window + 1)
        
        # Combine chunks in context window
        context_chunks = self.chunks[start_idx:end_idx]
        extended_text = ' '.join(context_chunks)
        
        # Create extended timestamp
        start_timestamp = self.timestamps[start_idx]
        end_timestamp = self.timestamps[end_idx - 1]
        
        start_time = start_timestamp.split(' - ')[0]
        end_time = end_timestamp.split(' - ')[1]
        extended_timestamp = f"{start_time} - {end_time}"
        
        return extended_timestamp, extended_text

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model:
            return {
                'model_name': self.model_name,
                'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 'Unknown',
                'num_chunks': len(self.chunks)
            }
        return {}