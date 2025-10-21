import json
import requests
import faiss
import time
import tiktoken
from sentence_transformers import SentenceTransformer

API_KEY=""
API_URL=""
MODEL = "gpt-4"
MAX_MODEL_TOKENS = 4096
RESPONSE_TOKEN_BUFFER = 1800
MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - RESPONSE_TOKEN_BUFFER

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_list = [
    {"question": "How do I reset my password?",
     "answer": "To reset your password, go to the account settings page, click on 'Forgot Password,' and follow the instructions sent to your registered email."},
    {"question": "How do I update my email address?",
     "answer": "Go to your profile settings, choose 'Email', and enter your new email. Verify it via the confirmation email."},
    {"question": "What should I do if my account is locked?",
     "answer": "Contact customer support or use the account recovery form to unlock your account."},
    {"question": "What are the password requirements?",
     "answer": "The password must be atleast 8 characters long with one special character, one capital letter and one numeric value."},
    {"question": "How can I delete my account permanently?",
     "answer": "Visit account settings, scroll down to 'Delete Account', and follow the confirmation steps. Note: this action is irreversible."}
]

def count_tokens(text, model_name=MODEL):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def truncate_to_token_limit(text, max_tokens, model_name=MODEL):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])
    return truncated

def call_wrapper_api(system_prompt, user_prompt, rate_limit=3):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": MODEL,
        "temperature": 0.5,
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
            return data['choices'][0]['message']['content'].strip()
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
            time.sleep(2)
    return None

def best_faq_match_embedding(expanded_query, faq_list, model):
    faq_questions = [faq['question'] for faq in faq_list]
    faq_embeddings = model.encode(faq_questions).astype('float32')
    dim = faq_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(faq_embeddings)
    query_embedding = model.encode([expanded_query]).astype('float32')
    D, I = index.search(query_embedding, 1)
    best_idx = int(I[0][0])
    return faq_list[best_idx]

if __name__ == "__main__":
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("Input cannot be empty. Exiting.")
        exit(1)

    system_prompt_expand = "Rewrite the user's query to include synonyms and related terms."
    total_tokens = count_tokens(system_prompt_expand, MODEL) + count_tokens(user_query, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Input is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_expand, MODEL)
        user_query = truncate_to_token_limit(user_query, allowed_user_tokens, MODEL)

    expanded_query = call_wrapper_api(system_prompt_expand, user_query)
    if not expanded_query:
        print("Query expansion failed.")
        exit(1)
    print("\n*** Expanded Query ***")
    print(expanded_query)

    matched_faq = best_faq_match_embedding(expanded_query, faq_list, model)
    if not matched_faq:
        print("No FAQ matched.")
        exit(1)

    print("\n*** Retrieved FAQ ***")
    print(f"Q: {matched_faq['question']}")
    print(f"A: {matched_faq['answer']}")

    system_prompt_answer = "You are a helpful assistant. Create a tailored answer using the user's query and the selected FAQ."
    tailoring_prompt = f"User Query: {user_query}\n\nFAQ:\nQ: {matched_faq['question']}\nA: {matched_faq['answer']}"

    # Token count and truncation for tailoring prompt
    total_tokens = count_tokens(system_prompt_answer, MODEL) + count_tokens(tailoring_prompt, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Tailoring prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_tailoring_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_answer, MODEL)
        tailoring_prompt = truncate_to_token_limit(tailoring_prompt, allowed_tailoring_tokens, MODEL)

    final_answer = call_wrapper_api(system_prompt_answer, tailoring_prompt)

    print("\n*** Final Answer ***")
    print(final_answer if final_answer else "Failed to generate final answer.")

