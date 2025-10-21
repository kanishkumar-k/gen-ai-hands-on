import json
import os
import requests
import numpy as np
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer

SNIPPETS_FILE = "snippet_metadata.json"
API_KEY=""
API_URL=""
MODEL = "gpt-4"
MAX_MODEL_TOKENS = 4096
RESPONSE_TOKEN_BUFFER = 1800
MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - RESPONSE_TOKEN_BUFFER

model = SentenceTransformer('all-MiniLM-L6-v2')

def count_tokens(text, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_to_token_limit(text, max_tokens, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

class VectorStore:
    def __init__(self, meta_file=SNIPPETS_FILE):
        self.meta_file = meta_file
        self.snippets = []
        self.index = None
        self.load_and_build_index()

    def load_and_build_index(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.snippets = json.load(f)
        else:
            self.snippets = []
        if self.snippets:
            codes = [s["code"] for s in self.snippets]
            embeddings = model.encode(codes).astype(np.float32)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index = None

    def query(self, query_text, top_k=2):
        if not self.snippets or self.index is None or self.index.ntotal == 0:
            print("No snippets in vector store.")
            return []
        query_emb = model.encode([query_text])[0].astype(np.float32)
        D, I = self.index.search(np.expand_dims(query_emb, axis=0), top_k)
        return [self.snippets[i] for i in I[0] if i < len(self.snippets)]

def call_wrapper_api(system_prompt: str, user_prompt: str, rate_limit: int = 3) -> str:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": MODEL,
        "temperature": 0.6,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": RESPONSE_TOKEN_BUFFER
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
            print(f"Retrying, Making an LLM request again ({trials + 1}/{rate_limit})")
            import time; time.sleep(2)
    return None

def main():
    vector_store = VectorStore()
    user_req = input("Describe your code requirement:\n").strip()
    if not user_req:
        print("Requirement cannot be empty. Exiting.")
        return

    system_prompt_1 = (
        "You are a helpful Python assistant. Given a user requirement and code references, "
        "generate boilerplate code that solves the requirement and incorporates best practices from the references."
    )

    relevant_snippets = vector_store.query(user_req, top_k=2)

    print("\n--- Retrieved Code References ---\n")
    for idx, snip in enumerate(relevant_snippets, 1):
        desc = snip.get("description", "")
        print(f"Snippet {idx} ({snip['name']}): {desc}\nCode:\n{snip['code']}\n")

    user_prompt_1 = (
        f"Requirement:\n{user_req}\n\n"
        "Relevant code references:\n"
        + "\n\n".join([snip["code"] for snip in relevant_snippets])
        + "\n\nGenerate the initial code solution."
    )
    total_tokens = count_tokens(system_prompt_1) + count_tokens(user_prompt_1)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_1)
        user_prompt_1 = truncate_to_token_limit(user_prompt_1, allowed_user_tokens)

    initial_code = call_wrapper_api(system_prompt_1, user_prompt_1)
    print("\n--- Initial Generated Code ---\n")
    print(initial_code if initial_code else "Failed to generate code.")

    if not initial_code:
        return

    system_prompt_2 = (
        "You are a senior Python engineer. Refactor the provided code for clarity, efficiency. Do not add any comments or explanations. "
        "Only return the improved function block and function call, nothing else is needed. Do not include ```python or ``` tags."
    )
    user_prompt_2 = f"Here is the code to refactor:\n\n{initial_code}"
    total_tokens = count_tokens(system_prompt_2) + count_tokens(user_prompt_2)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Refactor prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_2)
        user_prompt_2 = truncate_to_token_limit(user_prompt_2, allowed_user_tokens)

    final_code = call_wrapper_api(system_prompt_2, user_prompt_2)
    print("\n--- Refactored Final Code ---\n")
    print(final_code if final_code else "Failed to refactor code.")

    if final_code:
        filename = "final_code.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_code)
        print(f"\nFinal code saved to {filename}")

if __name__ == "__main__":
    main()

