import json
import requests
import tiktoken

CATEGORIES = [ 
    "Entertainment", "Technology", "Fashion", "Finance", "Food", "Health", 
    "Education", "Politics", "Lifestyle", "Business", 
    "Environment", "Travel", "Sports", "Other"
]

API_KEY=""
API_URL=""
MODEL = "gpt-4"
MAX_MODEL_TOKENS = 4096
RESPONSE_TOKEN_BUFFER = 1800
MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - RESPONSE_TOKEN_BUFFER

def count_tokens(text, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_to_token_limit(text, max_tokens, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

def call_wrapper_api(system_prompt: str, user_prompt: str, rate_limit: int = 3) -> str:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": MODEL,
        "temperature": 0.8,
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
            return response.json()['choices'][0]['message']['content']
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

def get_input_text():
    print("Paste your document text below. Press Enter, then Ctrl+Z to finish\n")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()

if __name__ == "__main__":
    document_text = get_input_text()
    if not document_text:
        print("No document text provided. Exiting.")
        exit(1)

    # Step 1: Summarization OF user content
    system_prompt_1 = (
        "You are a professional summarizer. Summarize the following document into 4 to 6 clear bullet points. "
        "Be concise and focus only on key information."
    )

    total_tokens = count_tokens(system_prompt_1) + count_tokens(document_text)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Input is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_1)
        document_text = truncate_to_token_limit(document_text, allowed_user_tokens)

    summary = call_wrapper_api(system_prompt_1, document_text)

    if summary:
        print("\n==== Step 1: SUMMARY (Bullet Points) ====\n")
        print(summary)
    else:
        print("Summarization failed.")
        exit(1)

    # Step 2: Classification using predefined categories
    category_list = ", ".join(CATEGORIES)
    system_prompt_2 = (
        f"You are a content classifier. Based on the following summary, classify the content into ONE of these categories:\n"
        f"{category_list}\n"
        "Respond with only the most appropriate category name from the list. Do not invent new categories."
    )

    total_tokens = count_tokens(system_prompt_2) + count_tokens(summary)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Classification prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_2)
        summary = truncate_to_token_limit(summary, allowed_user_tokens)

    category = call_wrapper_api(system_prompt_2, summary)

    if category:
        print("\n==== Step 2: CATEGORY ====\n")
        print(f"Category: {category.strip()}")
    else:
        print("Classification failed.")
        exit(1)

    # Step 3a: Formal style
    system_prompt_3a = (
        "You are a corporate communications expert. Restyle the following summary in a formal, professional tone."
    )

    total_tokens = count_tokens(system_prompt_3a) + count_tokens(summary)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Formal style prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_3a)
        summary = truncate_to_token_limit(summary, allowed_user_tokens)

    formal_output = call_wrapper_api(system_prompt_3a, summary)

    if formal_output:
        print("\n===== Step 3a: FORMAL STYLE =====\n")
        print(formal_output)
    else:
        print("Formal restyling failed.")

    # Step 3b: Casual style
    system_prompt_3b = (
        "You are a friendly blogger. Restyle the following summary in a casual, conversational tone."
    )

    total_tokens = count_tokens(system_prompt_3b) + count_tokens(summary)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Casual style prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_3b)
        summary = truncate_to_token_limit(summary, allowed_user_tokens)

    casual_output = call_wrapper_api(system_prompt_3b, summary)

    if casual_output:
        print("\n===== Step 3b: CASUAL STYLE =====\n")
        print(casual_output)
    else:

        print("Casual restyling failed.")
