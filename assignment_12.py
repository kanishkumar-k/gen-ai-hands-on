import json
import requests
import faiss
import time
import tiktoken
from sentence_transformers import SentenceTransformer

API_URL = "https://openai-api-wrapper-urtjok3rza-wl.a.run.app/api/chat/completions/"
API_KEY = "eyJraWQiOiI1cHpvN3Z0NnBiKzB3OEtwS3kwN0VWUzgzYnozSTZuVmhVT1A3dTlpYnlzPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoiS2Nzc2wxTXJfVzZjejk1TjIzc1JTdyIsInN1YiI6IjJiYmY0YTVkLWFiMjMtNGM3My04ZDlkLWFhZjkyMTY0YjExNSIsImNvZ25pdG86Z3JvdXBzIjpbInVzLWVhc3QtMl9vSTk5c0RKT1RfRGVsb2l0dGUiXSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMi5hbWF6b25hd3MuY29tXC91cy1lYXN0LTJfb0k5OXNESk9UIiwiY3VzdG9tOm9yZ19pZCI6IkRlbG9pdHRlIiwiY3VzdG9tOmpvYl90aXRsZSI6IlNvZnR3YXJlIEVuZ2luZWVyIEkiLCJpZGVudGl0aWVzIjpbeyJ1c2VySWQiOiJOTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicHJvdmlkZXJOYW1lIjoiRGVsb2l0dGUiLCJwcm92aWRlclR5cGUiOiJPSURDIiwiaXNzdWVyIjpudWxsLCJwcmltYXJ5IjoidHJ1ZSIsImRhdGVDcmVhdGVkIjoiMTc0NDEwNTQ2MjY2OSJ9XSwiYXV0aF90aW1lIjoxNzUyMTI0NjgwLCJjdXN0b206ZW1haWxzIjoie1wicHJpbWFyeV9lbWFpbFwiOiBcImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb21cIiwgXCJzZWNvbmRhcnlfZW1haWxzXCI6IFtdfSIsImV4cCI6MTc1Mzk3ODU2MCwiaWF0IjoxNzUzOTc0OTYyLCJqdGkiOiJlYzliYTc3NS1lMjIwLTRlZjgtOTUyZC02MDMxZDE1MjRlZWYiLCJlbWFpbCI6ImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb20iLCJvcmdhbmlzYXRpb25fZGV0YWlscyI6Ilt7XCJuYW1lXCI6IFwiSGFzaGVkSW5cIiwgXCJ0ZW5hbnRfaWRcIjogXCIxMTIyXCIsIFwiYWxsb3dlZF9hcHBzXCI6IFtcImFsbG9jYXRpb25cIiwgXCJsZWF2ZXNcIiwgXCJyZXdhcmRzXCIsIFwiaHUtYXV0b21hdGlvblwiLCBcInRpbWVzaGVldFwiLCBcInB1bHNlXCIsIFwidm9sdW50ZWVyXCIsIFwiaG9tZVwiLCBcIm9wc01ldHJpY3NcIiwgXCJodS1ldmFsdWF0aW9uXCIsIFwiYXV0b2RpZGFjdFwiLCBcImt1ZG9zXCIsIFwicG9kc1wvcHVyc3VpdHNcIiwgXCJoaXJlXCIsIFwicG9kc1wiLCBcImN1bXVsdXNcIiwgXCJyZWxheVwiLCBcInJ0djNcIiwgXCJsZWFybmluZ25kZXZcIiwgXCJjYW1cIl0sIFwidGltZXpvbmVcIjogXCJHTVRcIiwgXCJjdXJyZW5jeVwiOiBcIklOUlwiLCBcImxvZ29fczNfb2JqZWN0X3VybFwiOiBcImh0dHBzOlwvXC9kbmEtcHJvZC1yZXNvdXJjZS5zMy51cy1lYXN0LTIuYW1hem9uYXdzLmNvbVwvb3JnX2RpcmVjdG9yeVwvb3JnYW5pc2F0aW9uX2xvZ29zXC9oYXNoZWRpbi5wbmdcIn1dIiwiY3VzdG9tOnV1aWQiOiI4NDVmYTI2MC00NDA0LTQxYjktYTIzNS0xYWFlNmMyNGE1NDIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImNvZ25pdG86dXNlcm5hbWUiOiJEZWxvaXR0ZV9OTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicGljdHVyZSI6Imh0dHBzOlwvXC9ncmFwaC5taWNyb3NvZnQuY29tXC92MS4wXC9tZVwvcGhvdG9cLyR2YWx1ZSIsIm9yaWdpbl9qdGkiOiJjODU2MzdhYi0yMTgyLTQzMDAtYTZmYS1jMTExOGI4ZTI1MDQiLCJhdWQiOiI1YnB0a3JzcnNtOWZkcHU5ODB2a2RoZ29vcCIsInRva2VuX3VzZSI6ImlkIiwibmFtZSI6IkthcnVuYWthcmFuLCBLYW5pc2hrdW1hciIsInByaW1hcnlfdGVuYW50IjoiMTEyMiIsInNlc3Npb25fdGVuYW50IjoiMTEyMiJ9.eUi9IfGgyOJsswZfy3lxtljN8CAjtkmuPRNUL4HB5n2yq-oNRvGB3vHL4YHL93c2rXRVKgEaoYxzm2RERtFy85dlhMiSdolm7QJNwGOIU3K7drfM9_ggltIJheg8WB_IXtBuNN6jp4TiBzfGrnePsJSjL5KpBQTR1iEABl5cr2-UvFJDRHae3sDk5G0q5Yq_NKSROGpRVI6cFc93ZeoeKFDiGVVxfXE1KmMU04K6pYjs4VcLAraqpHaRBfgz5SDOHlIL8xipryS6hwcbgRForD-WjMRuAvA6AFrh4zwDH_hAIZGPiYjjs0FBvf2xio1-QYocOkAeXruMgb3gJt8wZw"
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
