import os
import json
import requests
import time
import tiktoken
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "gpt-4"
API_KEY=""
API_URL=""
MAX_MODEL_TOKENS = 4096
RESPONSE_TOKEN_BUFFER = 1800
MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - RESPONSE_TOKEN_BUFFER

DOCUMENTS = {
    "Employee Handbook": "Employee_Handbook.pdf",
    "ACE Handbook": "ACE_Handbook.pdf",
}

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def pdf_to_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def build_document_chunks():
    doc_chunks = []
    for name, path in DOCUMENTS.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        chunks = pdf_to_chunks(path)
        for idx, chunk in enumerate(chunks):
            doc_chunks.append({
                "doc": name,
                "chunk_id": idx,
                "text": chunk
            })
    return doc_chunks

def build_faiss_index(chunks, embedding_model):
    texts = [chunk['text'] for chunk in chunks]
    if not texts:
        raise ValueError("No document chunks to embed.")
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.atleast_2d(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def find_relevant_chunks_semantic(question, doc_chunks, embedding_model, index, top_k=6):
    query_emb = embedding_model.encode([question], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    return [doc_chunks[i] for i in I[0]]

def count_tokens(text, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_to_token_limit(text, max_tokens, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])
    return truncated

def call_wrapper_api(system_prompt: str, user_prompt: str, rate_limit: int = 3) -> str:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": MODEL,
        "temperature": 0.2,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": RESPONSE_TOKEN_BUFFER,
    })

    headers = {
        'x-api-token': API_KEY,
        'Content-Type': 'application/json',
    }

    for trials in range(rate_limit):
        try:
            response = requests.post(API_URL, data=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
        except KeyError as key_err:
            print(f"Key error: missing expected data {key_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
        if trials < rate_limit - 1:
            print(f"Retrying ({trials + 1}/{rate_limit})")
            time.sleep(2)
    return None

def main():
    print("Loading and chunking documents...")
    doc_chunks = build_document_chunks()
    print(f"Loaded {len(doc_chunks)} chunks from {len(DOCUMENTS)} documents.")
    print("Embedding chunks and building FAISS index...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    index, _ = build_faiss_index(doc_chunks, embedding_model)

    system_prompt_template = (
        "You are an expert HR policy assistant. "
        "Answer the user's question using ONLY the provided excerpts from company policy documents. "
        "If the answer is not directly stated, synthesize and infer from all excerpts. "
        "If you still cannot answer, reply: 'I don't have this information.'\n\n"
        "Here are the relevant excerpts:\n{context}"
    )

    print("Ask a question about the policy documents (type 'exit' to quit):")
    while True:
        user_prompt = input("\nYour question: ").strip()
        if user_prompt.lower() == "exit":
            break
        if not user_prompt:
            print("Please enter a question.")
            continue

        relevant_chunks = find_relevant_chunks_semantic(user_prompt, doc_chunks, embedding_model, index, top_k=6)
        if not relevant_chunks:
            print("\nAnswer:\nI don't have this information.")
            continue

        context = "\n".join(
            f"--- {chunk['doc']} (chunk {chunk['chunk_id']}) ---\n{chunk['text']}" for chunk in relevant_chunks
        )
        context = truncate_to_token_limit(context, MAX_INPUT_TOKENS)
        system_prompt = system_prompt_template.format(context=context)
        answer = call_wrapper_api(system_prompt, user_prompt)
        if answer:
            print("\nAnswer:\n", answer.strip())
        else:
            print("\nAnswer:\nI don't have this information.")

if __name__ == "__main__":

    main()
