import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSearcher:
    def __init__(self, chunks, timestamps):
        self.chunks = chunks
        self.timestamps = timestamps
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index from transcript chunks"""
        print("Building TF-IDF index...")
        
        # Initialize TF-IDF vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            max_features=5000,   # Limit vocabulary size
            min_df=1,           # Minimum document frequency
            max_df=0.95,        # Maximum document frequency
            sublinear_tf=True   # Use sublinear tf scaling
        )
        
        # Fit and transform the transcript chunks
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            print(f"TF-IDF index built with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} features")
        except Exception as e:
            print(f"Error building TF-IDF index: {e}")
            raise
    
    def search(self, query, top_k=1, similarity_threshold=0.1):
        """
        Search for the most relevant chunk using TF-IDF and cosine similarity
        
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
            # Ensure the vectorizer is initialized
            if self.vectorizer is None or self.tfidf_matrix is None:
                raise ValueError("Vectorizer is not initialized. Call _build_index() first.")
            
            # Transform query using the fitted vectorizer
            if self.vectorizer is None or self.tfidf_matrix is None:
                self._build_index()
            
            query_vector = self.vectorizer.transform([query.lower()])
            
            # Check if query vector is empty
            # type: ignore[attr-defined]  # nnz is valid for scipy sparse matrices
            if query_vector.nnz == 0:  # type: ignore[attr-defined]
                print("Warning: Query contains no recognized terms")
                return None
            
            # Calculate cosine similarity between query and all documents
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
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
    
    def search_multiple(self, query, top_k=3, similarity_threshold=0.1):
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
            # Ensure the vectorizer is initialized
            if self.vectorizer is None or self.tfidf_matrix is None:
                self._build_index()
            
            # Transform query
            if self.vectorizer is None:
                raise ValueError("Vectorizer is not initialized. Call _build_index() first.")
            query_vector = self.vectorizer.transform([query.lower()])
            # Check if query vector is empty
            # type: ignore[attr-defined]  # nnz is valid for scipy sparse matrices
            if query_vector.nnz == 0:  # type: ignore[attr-defined]
                print("Warning: Query contains no recognized terms")
                return []
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
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
    
    def get_feature_names(self):
        """Get the feature names (terms) from the vectorizer"""
        if self.vectorizer:
            return self.vectorizer.get_feature_names_out()
        return []
    
    def explain_similarity(self, query, chunk_idx):
        """
        Explain why a particular chunk was similar to the query
        
        Args:
            query (str): The search query
            chunk_idx (int): Index of the chunk to explain
        
        Returns:
            dict: Explanation of similarity factors
        """
        try:
            # Ensure vectorizer is initialized
            if self.vectorizer is None or self.tfidf_matrix is None:
                raise ValueError("Vectorizer is not initialized.")
            
            # Validate chunk index
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                raise ValueError(f"Invalid chunk index: {chunk_idx}")
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Transform query
            query_vector = self.vectorizer.transform([query.lower()])
            # Check if query vector is empty
            # type: ignore[attr-defined]  # nnz is valid for scipy sparse matrices
            if query_vector.nnz == 0:  # type: ignore[attr-defined]
                return {
                    'error': 'Query contains no recognized terms',
                    'common_terms': [],
                    'total_similarity': 0.0
                }
            
            # Get document vector
            
            doc_vector = self.tfidf_matrix.getrow(chunk_idx)
            
            # Convert to dense arrays safely
            try:
                query_features = np.asarray(query_vector.todense()).flatten()
                doc_features = np.asarray(doc_vector.todense()).flatten()
            except Exception as e:
                print(f"Error converting to array: {e}")
                return {
                    'error': f'Array conversion failed: {e}',
                    'common_terms': [],
                    'total_similarity': 0.0
                }
            
            # Find overlapping terms
            common_terms = []
            for i, (q_score, d_score) in enumerate(zip(query_features, doc_features)):
                if q_score > 0 and d_score > 0:
                    common_terms.append({
                        'term': feature_names[i],
                        'query_score': float(q_score),
                        'doc_score': float(d_score),
                        'contribution': float(q_score * d_score)
                    })
            
            # Sort by contribution
            common_terms.sort(key=lambda x: x['contribution'], reverse=True)
            
            # Calculate total similarity
            total_similarity = float(cosine_similarity(query_vector, doc_vector)[0][0])
            
            return {
                'common_terms': common_terms[:10],  # Top 10 contributing terms
                'total_similarity': total_similarity,
                'query_terms_found': len([t for t in common_terms if t['query_score'] > 0]),
                'doc_terms_matched': len([t for t in common_terms if t['doc_score'] > 0])
            }
            
        except Exception as e:
            print(f"Error explaining similarity: {e}")
            return {
                'error': str(e),
                'common_terms': [],
                'total_similarity': 0.0
            }