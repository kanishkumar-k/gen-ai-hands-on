import json
import requests
import time
import tiktoken

API_URL = "https://openai-api-wrapper-urtjok3rza-wl.a.run.app/api/chat/completions/"
API_KEY = "eyJraWQiOiI1cHpvN3Z0NnBiKzB3OEtwS3kwN0VWUzgzYnozSTZuVmhVT1A3dTlpYnlzPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoiS2Nzc2wxTXJfVzZjejk1TjIzc1JTdyIsInN1YiI6IjJiYmY0YTVkLWFiMjMtNGM3My04ZDlkLWFhZjkyMTY0YjExNSIsImNvZ25pdG86Z3JvdXBzIjpbInVzLWVhc3QtMl9vSTk5c0RKT1RfRGVsb2l0dGUiXSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMi5hbWF6b25hd3MuY29tXC91cy1lYXN0LTJfb0k5OXNESk9UIiwiY3VzdG9tOm9yZ19pZCI6IkRlbG9pdHRlIiwiY3VzdG9tOmpvYl90aXRsZSI6IlNvZnR3YXJlIEVuZ2luZWVyIEkiLCJpZGVudGl0aWVzIjpbeyJ1c2VySWQiOiJOTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicHJvdmlkZXJOYW1lIjoiRGVsb2l0dGUiLCJwcm92aWRlclR5cGUiOiJPSURDIiwiaXNzdWVyIjpudWxsLCJwcmltYXJ5IjoidHJ1ZSIsImRhdGVDcmVhdGVkIjoiMTc0NDEwNTQ2MjY2OSJ9XSwiYXV0aF90aW1lIjoxNzUyMTI0NjgwLCJjdXN0b206ZW1haWxzIjoie1wicHJpbWFyeV9lbWFpbFwiOiBcImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb21cIiwgXCJzZWNvbmRhcnlfZW1haWxzXCI6IFtdfSIsImV4cCI6MTc1Mzk3ODU2MCwiaWF0IjoxNzUzOTc0OTYyLCJqdGkiOiJlYzliYTc3NS1lMjIwLTRlZjgtOTUyZC02MDMxZDE1MjRlZWYiLCJlbWFpbCI6ImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb20iLCJvcmdhbmlzYXRpb25fZGV0YWlscyI6Ilt7XCJuYW1lXCI6IFwiSGFzaGVkSW5cIiwgXCJ0ZW5hbnRfaWRcIjogXCIxMTIyXCIsIFwiYWxsb3dlZF9hcHBzXCI6IFtcImFsbG9jYXRpb25cIiwgXCJsZWF2ZXNcIiwgXCJyZXdhcmRzXCIsIFwiaHUtYXV0b21hdGlvblwiLCBcInRpbWVzaGVldFwiLCBcInB1bHNlXCIsIFwidm9sdW50ZWVyXCIsIFwiaG9tZVwiLCBcIm9wc01ldHJpY3NcIiwgXCJodS1ldmFsdWF0aW9uXCIsIFwiYXV0b2RpZGFjdFwiLCBcImt1ZG9zXCIsIFwicG9kc1wvcHVyc3VpdHNcIiwgXCJoaXJlXCIsIFwicG9kc1wiLCBcImN1bXVsdXNcIiwgXCJyZWxheVwiLCBcInJ0djNcIiwgXCJsZWFybmluZ25kZXZcIiwgXCJjYW1cIl0sIFwidGltZXpvbmVcIjogXCJHTVRcIiwgXCJjdXJyZW5jeVwiOiBcIklOUlwiLCBcImxvZ29fczNfb2JqZWN0X3VybFwiOiBcImh0dHBzOlwvXC9kbmEtcHJvZC1yZXNvdXJjZS5zMy51cy1lYXN0LTIuYW1hem9uYXdzLmNvbVwvb3JnX2RpcmVjdG9yeVwvb3JnYW5pc2F0aW9uX2xvZ29zXC9oYXNoZWRpbi5wbmdcIn1dIiwiY3VzdG9tOnV1aWQiOiI4NDVmYTI2MC00NDA0LTQxYjktYTIzNS0xYWFlNmMyNGE1NDIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImNvZ25pdG86dXNlcm5hbWUiOiJEZWxvaXR0ZV9OTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicGljdHVyZSI6Imh0dHBzOlwvXC9ncmFwaC5taWNyb3NvZnQuY29tXC92MS4wXC9tZVwvcGhvdG9cLyR2YWx1ZSIsIm9yaWdpbl9qdGkiOiJjODU2MzdhYi0yMTgyLTQzMDAtYTZmYS1jMTExOGI4ZTI1MDQiLCJhdWQiOiI1YnB0a3JzcnNtOWZkcHU5ODB2a2RoZ29vcCIsInRva2VuX3VzZSI6ImlkIiwibmFtZSI6IkthcnVuYWthcmFuLCBLYW5pc2hrdW1hciIsInByaW1hcnlfdGVuYW50IjoiMTEyMiIsInNlc3Npb25fdGVuYW50IjoiMTEyMiJ9.eUi9IfGgyOJsswZfy3lxtljN8CAjtkmuPRNUL4HB5n2yq-oNRvGB3vHL4YHL93c2rXRVKgEaoYxzm2RERtFy85dlhMiSdolm7QJNwGOIU3K7drfM9_ggltIJheg8WB_IXtBuNN6jp4TiBzfGrnePsJSjL5KpBQTR1iEABl5cr2-UvFJDRHae3sDk5G0q5Yq_NKSROGpRVI6cFc93ZeoeKFDiGVVxfXE1KmMU04K6pYjs4VcLAraqpHaRBfgz5SDOHlIL8xipryS6hwcbgRForD-WjMRuAvA6AFrh4zwDH_hAIZGPiYjjs0FBvf2xio1-QYocOkAeXruMgb3gJt8wZw"
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