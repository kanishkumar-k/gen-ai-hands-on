import json
import requests
import time
import tiktoken

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
    truncated = enc.decode(tokens[:max_tokens])
    return truncated

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
            print(f"Retrying, Making an LLM request again ({trials + 1}/{rate_limit})")
            time.sleep(2)
    return None

if __name__ == "__main__":
    system_prompt_summary = (
        "You are an expert in information synthesis. Summarize the following text, focusing on the main ideas and key points. "
        "Provide a concise summary that captures the essential information."
    )
    system_prompt_paraphrase = (
        "You are an expert at Paraphrasing. Paraphrase the following summary for a younger audience, using simple and casual language "
        "and relatable examples while keeping the original meaning intact."
    )

    print("Enter the text you want to summarize. Press Enter, then Ctrl+Z to finish: \n")
    user_input_lines = []
    try:
        while True:
            line = input()
            user_input_lines.append(line)
    except EOFError:
        pass

    user_prompt = "\n".join(user_input_lines).strip()

    if not user_prompt:
        print("Input cannot be empty. Exiting.")
        exit(1)

    # Token count and truncation
    total_tokens = count_tokens(system_prompt_summary, MODEL) + count_tokens(user_prompt, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Input is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_summary, MODEL)
        user_prompt = truncate_to_token_limit(user_prompt, allowed_user_tokens, MODEL)

    # Step 1: Summarize the user input
    summary = call_wrapper_api(system_prompt_summary, user_prompt)
    if summary:
        print("\n*** Summary ***\n")
        print(summary)

        # Step 2: Paraphrase the generated summary provided by wrapper API
        paraphrased = call_wrapper_api(system_prompt_paraphrase, summary)
        if paraphrased:
            print("\n*** Paraphrased Summary ***\n")
            print(paraphrased)
        else:
            print("Failed to paraphrase the summary.")
    else:
        print("Failed to summarize the input.")

