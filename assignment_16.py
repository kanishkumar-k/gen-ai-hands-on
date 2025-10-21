import json
import requests
import tiktoken

CATEGORIES = [ 
    "Entertainment", "Technology", "Fashion", "Finance", "Food", "Health", 
    "Education", "Politics", "Lifestyle", "Business", 
    "Environment", "Travel", "Sports", "Other"
]

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