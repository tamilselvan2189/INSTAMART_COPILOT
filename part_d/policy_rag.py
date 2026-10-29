"""
Policy RAG (Retrieval-Augmented Generation) Module

Implements retrieval-augmented generation for policy questions:
- Task 1: Document chunking with configurable overlap
- Task 2: TF-IDF based retrieval
- Task 3: Grounded question answering
- Task 4: Evaluation on policy_eval.csv

Dependencies: mock_llm, sklearn, pandas, numpy, pathlib
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mock_llm import chat


@dataclass
class ChunkInfo:
    """Represents a single chunk from a policy document."""
    source: str
    text: str
    word_start: int
    word_end: int


def chunk_documents(
    policies_dir: str = "data/policies",
    overlap_words: int = 10,
    chunk_size: int = 40
) -> List[Dict]:
    """
    Split policy documents into overlapping chunks.
    
    Args:
        policies_dir: Directory containing policy .txt files
        overlap_words: Number of words to overlap between consecutive chunks
        chunk_size: Target number of words per chunk
    
    Returns:
        List of dicts: [{"source": filename, "text": chunk_text, "word_start": i, "word_end": j}, ...]
    
    Design:
    - TF (Term Frequency): Measures how often a word appears in a document.
      Higher frequency → higher weight in TF-IDF score.
    - IDF (Inverse Document Frequency): Measures how rare a word is across all documents.
      Rare words → higher weight (better discrimination).
    - TF-IDF Limitation: Words appearing in different phrasings (e.g., "refund" vs. "money back")
      receive different scores, potentially missing relevant chunks with synonyms.
    """
    chunks = []
    policies_path = Path(policies_dir)
    
    # Handle relative paths: try from current directory first, then from parent directories
    if not policies_path.is_absolute() and not policies_path.exists():
        # Try to find policies directory by walking up the directory tree
        for parent in Path.cwd().parents:
            candidate = parent / policies_dir
            if candidate.exists():
                policies_path = candidate
                break
    
    # Process all .txt files in policies directory
    for policy_file in sorted(policies_path.glob("*.txt")):
        # Read policy document
        with open(policy_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into words
        words = text.split()
        
        # Create overlapping chunks
        chunk_start = 0
        while chunk_start < len(words):
            chunk_end = min(chunk_start + chunk_size, len(words))
            
            # Extract chunk text
            chunk_text = " ".join(words[chunk_start:chunk_end])
            
            # Record chunk with source and word positions
            chunks.append({
                "source": policy_file.name,
                "text": chunk_text,
                "word_start": chunk_start,
                "word_end": chunk_end
            })
            
            # Move to next chunk: advance by (chunk_size - overlap_words)
            # This ensures overlap_words of overlap between consecutive chunks
            chunk_start += (chunk_size - overlap_words)
            
            # Stop if we've reached the end of the document
            if chunk_end == len(words):
                break
    
    return chunks


def build_retriever(chunks: List[Dict]) -> Tuple[TfidfVectorizer, List[Dict]]:
    """
    Fit TF-IDF vectorizer on chunk texts.
    
    Args:
        chunks: List of chunk dicts from chunk_documents()
    
    Returns:
        Tuple: (fitted_vectorizer, chunks)
    """
    chunk_texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(chunk_texts)
    return vectorizer, chunks


def retrieve(
    query: str,
    k: int = 3,
    vectorizer: TfidfVectorizer = None,
    chunks: List[Dict] = None
) -> List[Dict]:
    """
    Retrieve top-k chunks most similar to query using cosine similarity.
    
    Args:
        query: User's query string
        k: Number of top chunks to return
        vectorizer: Fitted TfidfVectorizer
        chunks: List of chunk dicts
    
    Returns:
        List of dicts: [{"source": str, "score": float, "text": str}, ...]
        sorted by score (descending)
    """
    if vectorizer is None or chunks is None:
        raise ValueError("vectorizer and chunks must be provided")
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Transform all chunks
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_vectors = vectorizer.transform(chunk_texts)
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Build results
    results = []
    for idx in top_indices:
        results.append({
            "source": chunks[idx]["source"],
            "score": float(similarities[idx]),
            "text": chunks[idx]["text"]
        })
    
    return results


def answer(
    question: str,
    vectorizer: TfidfVectorizer = None,
    chunks: List[Dict] = None
) -> Tuple[str, List[str]]:
    """
    Answer a question grounded in retrieved policy context.
    
    Args:
        question: Question to answer
        vectorizer: Fitted TfidfVectorizer
        chunks: List of chunk dicts
    
    Returns:
        Tuple: (answer_text, list_of_sources)
    """
    if vectorizer is None or chunks is None:
        raise ValueError("vectorizer and chunks must be provided")
    
    # Retrieve relevant chunks
    retrieved = retrieve(question, k=3, vectorizer=vectorizer, chunks=chunks)
    
    # Format context for LLM
    context_parts = []
    sources = []
    for item in retrieved:
        context_parts.append(f"[{item['source']}]\n{item['text']}")
        if item['source'] not in sources:
            sources.append(item['source'])
    
    context_text = "\n\n".join(context_parts)
    
    # Build system message with "POLICY ASSISTANT" trigger
    system_message = {
        "role": "system",
        "content": (
            "POLICY ASSISTANT\n\n"
            "You are a policy assistant. Answer the user's question using ONLY the provided context. "
            "If the answer cannot be found in the context, respond with: 'I could not find the answer in the provided policies.'"
        )
    }
    
    # Build user message
    user_message = {
        "role": "user",
        "content": (
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{question}"
        )
    }
    
    # Call LLM
    response = chat([system_message, user_message])
    answer_text = response.choices[0].message.content
    
    return answer_text, sources


def evaluate_retrieval(
    overlap_words: int = 10,
    chunk_size: int = 40,
    eval_csv: str = "data/raw/policy_eval.csv",
    policies_dir: str = "data/policies"
) -> Dict:
    """
    Evaluate retrieval and grounding accuracy on policy_eval.csv.
    
    Evaluation logic:
    - For answerable questions (expected_doc != "NONE"):
      * Retrieval success: is expected_doc in top 3 retrieved chunks?
      * Grounding success: does answer contain expected_fact (case-sensitive)?
    - For unanswerable questions (expected_doc == "NONE"):
      * Fallback success: does answer start with "I could not find"?
    
    Args:
        overlap_words: Overlap parameter for chunking
        chunk_size: Chunk size parameter for chunking
        eval_csv: Path to evaluation CSV
        policies_dir: Path to policies directory
    
    Returns:
        Dict with results: {
            "overlap": overlap_words,
            "chunk_size": chunk_size,
            "retrieval_accuracy": float (0.0-1.0),
            "grounding_accuracy": float (0.0-1.0),
            "answerable_count": int,
            "unanswerable_count": int,
            "details": [
                {
                    "question": str,
                    "expected_doc": str,
                    "expected_fact": str,
                    "retrieval_success": bool,
                    "grounding_success": bool,
                    "answer": str,
                    "retrieved_sources": [str]
                },
                ...
            ]
        }
    """
    # Handle relative paths for eval_csv
    eval_csv_path = Path(eval_csv)
    if not eval_csv_path.is_absolute() and not eval_csv_path.exists():
        for parent in Path.cwd().parents:
            candidate = parent / eval_csv
            if candidate.exists():
                eval_csv_path = candidate
                break
    
    # Handle relative paths for policies_dir
    policies_path = Path(policies_dir)
    if not policies_path.is_absolute() and not policies_path.exists():
        for parent in Path.cwd().parents:
            candidate = parent / policies_dir
            if candidate.exists():
                policies_path = candidate
                break
    # Build chunks and retriever
    chunks = chunk_documents(
        policies_dir=str(policies_path),
        overlap_words=overlap_words,
        chunk_size=chunk_size
    )
    vectorizer, _ = build_retriever(chunks)
    
    # Load evaluation CSV
    eval_df = pd.read_csv(str(eval_csv_path))
    
    results_detail = []
    retrieval_hits = 0
    grounding_hits = 0
    answerable_count = 0
    unanswerable_count = 0
    
    for idx, row in eval_df.iterrows():
        question = row['question']
        expected_doc = row['expected_doc']
        expected_fact = row['expected_fact']
        
        # Get answer
        answer_text, sources = answer(question, vectorizer=vectorizer, chunks=chunks)
        
        # Evaluate
        retrieval_success = False
        grounding_success = False
        
        if expected_doc == "NONE":
            # Unanswerable question: check for fallback
            unanswerable_count += 1
            grounding_success = answer_text.startswith("I could not find")
            if grounding_success:
                grounding_hits += 1
        else:
            # Answerable question
            answerable_count += 1
            
            # Check retrieval: is expected_doc in retrieved sources?
            retrieval_success = expected_doc in sources
            if retrieval_success:
                retrieval_hits += 1
            
            # Check grounding: does answer contain expected_fact (case-sensitive)?
            grounding_success = expected_fact in answer_text
            if grounding_success:
                grounding_hits += 1
        
        results_detail.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact if expected_doc != "NONE" else None,
            "retrieval_success": retrieval_success if expected_doc != "NONE" else None,
            "grounding_success": grounding_success,
            "answer": answer_text,
            "retrieved_sources": sources
        })
    
    # Compute accuracies
    retrieval_accuracy = retrieval_hits / answerable_count if answerable_count > 0 else 0.0
    grounding_accuracy = grounding_hits / len(eval_df) if len(eval_df) > 0 else 0.0
    
    return {
        "overlap": overlap_words,
        "chunk_size": chunk_size,
        "retrieval_accuracy": retrieval_accuracy,
        "grounding_accuracy": grounding_accuracy,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "details": results_detail
    }
