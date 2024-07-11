import torch
import re
from collections import defaultdict
from transformers import pipeline, set_seed

# Check for CUDA availability and set seed for reproducibility
device = 0 if torch.cuda.is_available() else -1  
set_seed(42)  

# Initialize pipelines
summarizer = pipeline("summarization", model="t5-small", device=device)
interpreter = pipeline("text-generation", model="gpt2", device=device)
chatgpt = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=device, pad_token_id=50256)
sentiment_analyzer = pipeline("sentiment-analysis", device=device)

# Function to preprocess text
def preprocess_text(text):
    # Tokenization and normalization
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize by words and convert to lowercase
    return tokens

# Function to build inverted index with context
def build_inverted_index_with_context(text, window_size=5):
    inverted_index = defaultdict(list)
    tokens = preprocess_text(text)
    
    # Convert text to sentences for better context retrieval
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Build a mapping from token positions to sentence positions
    token_to_sentence = []
    token_count = 0
    for i, sentence in enumerate(sentences):
        sentence_tokens = preprocess_text(sentence)
        for _ in sentence_tokens:
            token_to_sentence.append(i)
        token_count += len(sentence_tokens)
    
    # Ensure the mapping is correct
    assert len(tokens) == token_count, "Mismatch between tokens and token-to-sentence mapping."
    
    # Build inverted index with context
    for position, token in enumerate(tokens):
        sentence_index = token_to_sentence[position]
        start = max(0, sentence_index - window_size)
        end = min(len(sentences), sentence_index + window_size + 1)
        context = " ".join(sentences[start:end])
        
        inverted_index[token].append({
            'position': sentence_index + 1,  # Adjust position as needed
            'context': context,
            'full_sentence': sentences[sentence_index]  # Full sentence where the token is found
        })
    
    return inverted_index

# Function to process text in chunks
def process_text_in_chunks(text, chunk_size=1000):
    chunks = []
    current_chunk = ""
    for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text):
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to search quotes in text
def search_quotes(text, query, summarizer, interpreter, chatgpt, sentiment_analyzer):
    index = build_inverted_index_with_context(text)
    if query in index:
        occurrences = index[query]
        results = []
        for occurrence in occurrences:
            # Use AI summarization to generate a summary of context
            summarized_context = summarize_text(occurrence['context'], occurrence['full_sentence'], summarizer)
            
            # Use AI text generation to interpret the meaning of the quote
            interpretation = generate_interpretation(occurrence['full_sentence'], interpreter)
            
            # Use ChatGPT-3 to explain the quote within the context of the whole book
            explanation = explain_quote(occurrence['full_sentence'], text, chatgpt)
            
            # Use sentiment analysis to analyze sentiment of the quote
            sentiment = analyze_sentiment(occurrence['full_sentence'], sentiment_analyzer)
            
            results.append({
                'quote': query,
                'position': occurrence['position'],
                'context_summary': summarized_context,
                'full_sentence': occurrence['full_sentence'],
                'interpretation': interpretation,
                'explanation': explanation,
                'sentiment': sentiment
            })
        return results
    else:
        return None

# Function to summarize text using AI
def summarize_text(context, full_sentence, summarizer):
    # Determine appropriate max_length dynamically based on input size
    input_length = len(context.split())
    max_length = min(200, input_length + 50)  # Adjusted for potential model constraints
    summarized_text = summarizer(context, max_length=max_length, min_length=10, do_sample=False, truncation=True)
    return summarized_text[0]['summary_text'] + " " + full_sentence

# Function to generate interpretation of the quote using AI (text generation)
def generate_interpretation(quote, interpreter):
    return interpreter(quote, max_length=100, do_sample=False)[0]['generated_text']

# Function to explain a quote within the context of the whole book using ChatGPT-3
def explain_quote(quote, book_text, chatgpt):
    input_text = f"Question: Explain the significance of the quote '{quote}' within the context of the book.\nContext: {book_text}"
    max_length = chatgpt.model.config.max_position_embeddings
    if len(input_text.split()) > max_length:
        input_text = " ".join(input_text.split()[:max_length])
    response = chatgpt(input_text, num_return_sequences=1, max_new_tokens=100)
    return response[0]['generated_text']

# Function to analyze sentiment of a quote using AI
def analyze_sentiment(sentence, sentiment_analyzer):
    sentiment = sentiment_analyzer(sentence)[0]
    label = sentiment['label']
    score = sentiment['score']
    return f"Sentiment: {label} (Score: {score:.2f})"

if __name__ == "__main__":
    # Example: Load text from "Catcher in the Rye" (replace with actual text file path)
    with open('the-catcher-rye.txt', 'r', encoding='utf-8') as file:
        book_text = file.read()
    
    # Process text in chunks
    text_chunks = process_text_in_chunks(book_text)
    
    # Example queries
    queries = ["phony"]
    
    for query in queries:
        print(f"Searching for '{query}' in 'Catcher in the Rye':")
        results = []
        for chunk in text_chunks:
            chunk_results = search_quotes(chunk, query, summarizer, interpreter, chatgpt, sentiment_analyzer)
            if chunk_results:
                results.extend(chunk_results)
        
        if results:
            for result in results:
                print(f"Quote found: '{result['quote']}'")
                print(f"Position: {result['position']}")
                print(f"Full Sentence: '{result['full_sentence']}'")
                print(f"Context Summary: {result['context_summary']}")
                print(f"Interpretation: {result['interpretation']}")
                print(f"Explanation: {result['explanation']}")
                print(f"{result['sentiment']}\n")
        else:
            print(f"'{query}' not found in 'Catcher in the Rye'.\n")
