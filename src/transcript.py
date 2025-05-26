import sys
import os
from utils import TranscriptLoader
from search_tfidf import TFIDFSearcher
from search_hf import HuggingFaceSearcher

def main():
    if len(sys.argv) != 3:
        print("Usage: python transcript.py <transcript_file> <method>")
        print("Methods: tfidf, llm2")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    method = sys.argv[2].lower()
    
    # Validation of the method
    if method not in ['tfidf', 'llm2']:
        print("Error: Method must be one of: tfidf, llm2")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(transcript_file):
        print(f"Error: File '{transcript_file}' not found")
        sys.exit(1)
    
    # Load transcript
    loader = TranscriptLoader(transcript_file)
    chunks, timestamps = loader.load_and_chunk()
    
    if not chunks:
        print("Error: No valid transcript chunks loaded. Check file format.")
        sys.exit(1)
    
    # Initialize appropriate searcher
    searcher = None
    if method == 'tfidf':
        searcher = TFIDFSearcher(chunks, timestamps)
    elif method == 'llm2':
        searcher = HuggingFaceSearcher(chunks, timestamps)
    
    if searcher is None:
        print("Error: Failed to initialize searcher.")
        sys.exit(1)
    
    print("Transcript loaded, please ask your question (press 8 for exit):")
    
    # Create output directory
    output_file_path = 'output/output.txt'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Interactive loop
    try:
        with open(output_file_path, 'a', encoding='utf-8') as out_file:
            while True:
                query = ""
                try:
                    query = input("> ").strip()
                    
                    if query == '8':
                        print("Semantic search ended.")
                        break
                    
                    if not query:
                        print("Please enter a valid question.")
                        continue
                    
                    # Search and display result
                    result = searcher.search(query)
                    if result:
                        timestamp, text = result
                        result_display = f"[{timestamp}], {text}"
                        print(result_display)
                    else:
                        result_display = "No relevant answer found."
                        print(result_display)
                    
                    # Log result in background
                    log_output = f"Question: {query}\nMethod: {method}\nOutput: {result_display}\n\n"
                    out_file.write(log_output)
                    
                except Exception as e:
                    print(f"Error processing query: {e}")
                    log_output = f"Question: {query}\nMethod: {method}\nOutput: Error - {e}\n\n"
                    out_file.write(log_output)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except IOError as e:
        print(f"Error writing to output file: {e}")
        log_output = f"Question: None\nMethod: {method}\nOutput: Error - {e}\n\n"
        with open(output_file_path, 'a', encoding='utf-8') as out_file:
            out_file.write(log_output)

if __name__ == "__main__":
    main()