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
            time.sleep(2)
    return None

if __name__ == "__main__":
    print("Paste the schema definition. Press Enter, then Ctrl+Z to finish:\n")
    schema_lines = []
    try:
        while True:
            line = input()
            schema_lines.append(line)
    except EOFError:
        pass
    schema = "\n".join(schema_lines).strip()

    if not schema:
        print("Schema input cannot be empty. Exiting.")
        exit(1)

    print("\nAsk any question in natural language. Press Enter, then Ctrl+Z to finish:\n")
    question_lines = []
    try:
        while True:
            line = input()
            question_lines.append(line)
    except EOFError:
        pass
    question = " ".join(question_lines).strip()

    if not question:
        print("Question input cannot be empty. Exiting.")
        exit(1)

    # Token count and truncation for schema + question
    system_prompt_sql = "You are an expert SQL assistant. Given a database schema and a user's question, generate a valid SQL query."
    user_prompt_sql = (
        f"Schema:\n{schema}\n\n"
        f"User Question:\n{question}\n\n"
        "Generate a valid SQL query that answers the user's question."
    )
    total_tokens = count_tokens(system_prompt_sql, MODEL) + count_tokens(user_prompt_sql, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Input is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_sql, MODEL)
        user_prompt_sql = truncate_to_token_limit(user_prompt_sql, allowed_user_tokens, MODEL)

    # Step 1: Print SQL Schema
    print("\n*** Schema ***\n")
    print(schema + "\n")

    # Step 2: Generate SQL
    initial_sql = call_wrapper_api(system_prompt_sql, user_prompt_sql)
    if initial_sql:
        print("\n*** Initial SQL Query ***\n")
        print(initial_sql)
    else:
        print("Failed to generate SQL.")
        exit(1)

    # Step 3: Validate and Refine SQL
    system_prompt_refine = "You are a meticulous SQL reviewer. Review and refine the SQL query if there are potential issues based on the provided schema and user question."
    user_prompt_refine = (
        f"Schema:\n{schema}\n\n"
        f"User Question:\n{question}\n\n"
        f"SQL Query:\n{initial_sql}\n\n"
        "Review the SQL for syntax and logical accuracy. Refine if needed and output the final SQL query only."
    )
    total_tokens = count_tokens(system_prompt_refine, MODEL) + count_tokens(user_prompt_refine, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Refinement prompt is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_refine, MODEL)
        user_prompt_refine = truncate_to_token_limit(user_prompt_refine, allowed_user_tokens, MODEL)

    refined_sql = call_wrapper_api(system_prompt_refine, user_prompt_refine)
    if refined_sql:
        print("\n*** Refined Final SQL Query ***\n")
        print(refined_sql)
    else:

        print("Failed to refine SQL.")
