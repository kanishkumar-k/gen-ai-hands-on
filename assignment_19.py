import os
import json
import requests
import time
import PyPDF2
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer, CrossEncoder

API_URL = "https://openai-api-wrapper-urtjok3rza-wl.a.run.app/api/chat/completions/"
API_KEY = "eyJraWQiOiI1cHpvN3Z0NnBiKzB3OEtwS3kwN0VWUzgzYnozSTZuVmhVT1A3dTlpYnlzPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoiaV9MVnEyN1lVQ20xTnpWdWlJX0FiZyIsInN1YiI6IjJiYmY0YTVkLWFiMjMtNGM3My04ZDlkLWFhZjkyMTY0YjExNSIsImNvZ25pdG86Z3JvdXBzIjpbInVzLWVhc3QtMl9vSTk5c0RKT1RfRGVsb2l0dGUiXSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMi5hbWF6b25hd3MuY29tXC91cy1lYXN0LTJfb0k5OXNESk9UIiwiY3VzdG9tOm9yZ19pZCI6IkRlbG9pdHRlIiwiY3VzdG9tOmpvYl90aXRsZSI6IlNvZnR3YXJlIEVuZ2luZWVyIEkiLCJpZGVudGl0aWVzIjpbeyJ1c2VySWQiOiJOTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicHJvdmlkZXJOYW1lIjoiRGVsb2l0dGUiLCJwcm92aWRlclR5cGUiOiJPSURDIiwiaXNzdWVyIjpudWxsLCJwcmltYXJ5IjoidHJ1ZSIsImRhdGVDcmVhdGVkIjoiMTc0NDEwNTQ2MjY2OSJ9XSwiYXV0aF90aW1lIjoxNzU0Mzc0Nzk5LCJjdXN0b206ZW1haWxzIjoie1wicHJpbWFyeV9lbWFpbFwiOiBcImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb21cIiwgXCJzZWNvbmRhcnlfZW1haWxzXCI6IFtdfSIsImV4cCI6MTc1NDQ2NzI5NCwiaWF0IjoxNzU0NDYzNjk1LCJqdGkiOiJhMGYyZWIzMC00ZTI1LTQzODktOGVmNC1mMDU0MTk0NDgyZmIiLCJlbWFpbCI6ImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb20iLCJvcmdhbmlzYXRpb25fZGV0YWlscyI6Ilt7XCJuYW1lXCI6IFwiSGFzaGVkSW5cIiwgXCJ0ZW5hbnRfaWRcIjogXCIxMTIyXCIsIFwiYWxsb3dlZF9hcHBzXCI6IFtcImFsbG9jYXRpb25cIiwgXCJsZWF2ZXNcIiwgXCJyZXdhcmRzXCIsIFwiaHUtYXV0b21hdGlvblwiLCBcInRpbWVzaGVldFwiLCBcInB1bHNlXCIsIFwidm9sdW50ZWVyXCIsIFwiaG9tZVwiLCBcIm9wc01ldHJpY3NcIiwgXCJodS1ldmFsdWF0aW9uXCIsIFwiYXV0b2RpZGFjdFwiLCBcImt1ZG9zXCIsIFwicG9kc1wvcHVyc3VpdHNcIiwgXCJoaXJlXCIsIFwicG9kc1wiLCBcImN1bXVsdXNcIiwgXCJyZWxheVwiLCBcInJ0djNcIiwgXCJsZWFybmluZ25kZXZcIiwgXCJjYW1cIl0sIFwidGltZXpvbmVcIjogXCJHTVRcIiwgXCJjdXJyZW5jeVwiOiBcIklOUlwiLCBcImxvZ29fczNfb2JqZWN0X3VybFwiOiBcImh0dHBzOlwvXC9kbmEtcHJvZC1yZXNvdXJjZS5zMy51cy1lYXN0LTIuYW1hem9uYXdzLmNvbVwvb3JnX2RpcmVjdG9yeVwvb3JnYW5pc2F0aW9uX2xvZ29zXC9oYXNoZWRpbi5wbmdcIn1dIiwiY3VzdG9tOnV1aWQiOiI4NDVmYTI2MC00NDA0LTQxYjktYTIzNS0xYWFlNmMyNGE1NDIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImNvZ25pdG86dXNlcm5hbWUiOiJEZWxvaXR0ZV9OTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicGljdHVyZSI6Imh0dHBzOlwvXC9ncmFwaC5taWNyb3NvZnQuY29tXC92MS4wXC9tZVwvcGhvdG9cLyR2YWx1ZSIsIm9yaWdpbl9qdGkiOiI5ZDhlMzU1NS1iOTBjLTRlN2QtYTYwMS1jZjI3YzY5MzE3YjQiLCJhdWQiOiI1YnB0a3JzcnNtOWZkcHU5ODB2a2RoZ29vcCIsInRva2VuX3VzZSI6ImlkIiwibmFtZSI6IkthcnVuYWthcmFuLCBLYW5pc2hrdW1hciIsInByaW1hcnlfdGVuYW50IjoiMTEyMiIsInNlc3Npb25fdGVuYW50IjoiMTEyMiJ9.O0f-7bsibxgw-3I3ZfSjiEDTtPaVqXZO1sEH_xQCaRIes1ZFl71oWg-7XIlJDHh0c-GguH4eTZ4M-T9h-tLooutyvMyx5nFEreGXxWrJezRGGhW37Y43-06KF1AxkOtFGfXy8oPG_kpRBHbkbY6z8lrb7IFUTrhMROIFG5uyxf8ai6sJGxMN9iA4mXddLxvPQIItPPzGVye54ZLqfabYNI294UNdNvQMELpctVxh59w2Y4s87LBgblCDT0ek0-WZmQ9BtVIIIoqcbXMXDU9i7Rr97wRvB9cVRuNhBVmT3O9m2-AVaE-Ln3OoqL9iq71qqWJs2on1rXSXlc8KU9XfVg"
MODEL_PRIMARY = "gpt-4"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1000
MAX_TOKENS = 4096
RESPONSE_TOKENS = 800  
MAX_INPUT_TOKENS = MAX_TOKENS - RESPONSE_TOKENS

DOCUMENTS = {
    "Document1": "Assignment_docs/Document1.pdf",
    "Document2": "Assignment_docs/Document2.pdf",
}

def extract_section_chunks(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        full_text = "\n".join(page.extract_text() or '' for page in reader.pages)

    if not full_text.strip():
        return []

    sections = full_text.split("\n")
    chunks = []
    current_section = []
    current_title = "Untitled"

    for line in sections:
        if line.strip().startswith(tuple(str(n) for n in range(10))):
            if current_section:
                chunks.append({"section": current_title, "text": " ".join(current_section).strip()})
            current_title = line.strip()
            current_section = []
        else:
            current_section.append(line)

    if current_section:
        chunks.append({"section": current_title, "text": " ".join(current_section).strip()})
    return chunks


def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def rerank_chunks(query, chunks, cross_encoder, top_k=5):
    pairs = [(query, chunk['text']) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]


def search_similar_chunks(query, chunks, index, model, cross_encoder=None, top_k=10):
    if not chunks:
        return []
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    if cross_encoder:
        return rerank_chunks(query, top_chunks, cross_encoder, top_k=4)
    return top_chunks

def count_tokens(text):
    enc = tiktoken.encoding_for_model(MODEL_PRIMARY)
    return len(enc.encode(text))

def truncate(text, limit):
    enc = tiktoken.encoding_for_model(MODEL_PRIMARY)
    tokens = enc.encode(text)
    return enc.decode(tokens[:limit]) if len(tokens) > limit else text

def call_wrapper_api(system_prompt, user_prompt, model=MODEL_PRIMARY, rate_limit=3):
    headers = {
        'x-api-token': API_KEY,
        'Content-Type': 'application/json',
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": RESPONSE_TOKENS,
        "temperature": 0.2
    }
    for trial in range(rate_limit):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
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
        if trial < rate_limit - 1:
            print(f"Retrying, Making an LLM request again ({trial + 1}/{rate_limit})")
            time.sleep(2)
    return None

def compare_documents(question, doc1_chunks, doc2_chunks, embedding_model, cross_encoder, index1, index2):
    doc1_matches = search_similar_chunks(question, doc1_chunks, index1, embedding_model, cross_encoder, top_k=10)
    doc2_matches = search_similar_chunks(question, doc2_chunks, index2, embedding_model, cross_encoder, top_k=10)

    doc1_context = "\n\n".join([f"--- Document 1 | Section: {ch['section']} ---\n{ch['text']}" for ch in doc1_matches]) if doc1_matches else "No relevant content found in Document 1."
    doc2_context = "\n\n".join([f"--- Document 2 | Section: {ch['section']} ---\n{ch['text']}" for ch in doc2_matches]) if doc2_matches else "No relevant content found in Document 2."

    prompt = f"""
Based on the following excerpts from two documents, answer this question:
{question}

[Context Start]
{doc1_context}
{doc2_context}
[Context End]

Respond ONLY with the 3-4 most important context based on the question. Avoid repetition. No fluff.
Use clean bullet points.
"""
    system_prompt = (
        "You are a document comparison analyst. Use only the given content to generate a concise, structured answer."
    )
    return call_wrapper_api(system_prompt, truncate(prompt, MAX_INPUT_TOKENS), model=MODEL_PRIMARY)

def main():
    print("Loading and parsing documents...")
    doc1_chunks = extract_section_chunks(DOCUMENTS["Document1"])
    doc2_chunks = extract_section_chunks(DOCUMENTS["Document2"])


    if not doc1_chunks:
        print("No content extracted from Document 1")
    if not doc2_chunks:
        print("No content extracted from Document 2")


    print("Embedding documents...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    index1, _ = embed_chunks(doc1_chunks, embedding_model)
    index2, _ = embed_chunks(doc2_chunks, embedding_model)


    print("Loading reranking cross-encoder...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


    questions = [
        "How does the message structure for 'List of Participants by Discipline' in Cycling BMX Freestyle (Document 1) differ when applied to Cycling BMX Racing as described in Document 2?",
        "In Document 2, the Event Unit Start List and Results require certain triggers for BMX Racing. How would these triggers apply if adapted for BMX Freestyle as outlined in Document 1?",
        "How does the Event Final Ranking message in BMX Racing (Document 2) influence the format and data requirements for a similar message in BMX Freestyle (Document 1)?",
        "What are the specific ways the 'Applicable Messages' section for BMX Racing (Document 2) alters the permitted use of the Cycling BMX Freestyle Data Dictionary (Document 1)?",
        "How does the implementation of the 'ExtendedInfo' types in BMX Racing (Document 2) affect the development of BMX Freestyle standards based on guidelines in Document 1?"
    ]

    for idx, q in enumerate(questions, 1):
        print(f"\nQ{idx}: {q}\n")
        answer = compare_documents(q, doc1_chunks, doc2_chunks, embedding_model, cross_encoder, index1, index2)
        print(f"\n{answer.strip()}\n{'─'*80}")

if __name__ == "__main__":
    main()