import json
import requests
import subprocess
import os
import time
import tiktoken

VAPI_KEY=""
API_URL=""
PLANTUML_JAR_PATH = "plantuml.jar"
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

def render_plantuml_locally(plantuml_code, output_file="diagram.png", plantuml_jar_path="plantuml.jar"):
    temp_puml = "temp_diagram.puml"
    with open(temp_puml, "w", encoding="utf-8") as f:
        f.write(plantuml_code)
    try:
        subprocess.run(
            ["java", "-jar", plantuml_jar_path, "-tpng", temp_puml, "-o", "."],
            check=True
        )
        temp_png = "temp_diagram.png"
        if os.path.exists(temp_png):
            os.rename(temp_png, output_file)
        print(f"Diagram image saved as {output_file}")
    except Exception as e:
        print(f"Failed to generate image via local PlantUML: {e}")
    finally:
        try:
            os.remove(temp_puml)
        except Exception:
            pass

if __name__ == "__main__":
    system_prompt_parse = (
        "You are a system analyst. Parse the following user description of a system or process. "
        "Identify all components (entities, actors, or modules) and the data flows between them. "
        "Return a JSON object with two arrays: 'components' (list of unique component names) and 'flows' "
        "(list of objects with 'source', 'target', and optional 'label')."
    )
    system_prompt_plantuml = (
        "You are an expert in diagramming. Given a JSON object with 'components' and 'flows', "
        "generate a PlantUML diagram that accurately represents the data flows between components. "
        "Use 'entity' for each component. Use arrows for flows. Output only the PlantUML code."
    )
    system_prompt_validate = (
        "You are a PlantUML expert. Validate the following PlantUML code for syntax and logical errors. "
        "If there are issues, list them clearly. If valid, reply 'Valid'."
    )
    system_prompt_correct = (
        "You are a PlantUML expert. Correct the following PlantUML code to fix any syntax or logical errors. "
        "Return only the corrected PlantUML code."
    )

    print("Paste the system description. Press Enter, then Ctrl+Z to finish:\n")
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
    total_tokens = count_tokens(system_prompt_parse, MODEL) + count_tokens(user_prompt, MODEL)
    if total_tokens > MAX_INPUT_TOKENS:
        print(f"Input is too long ({total_tokens} tokens). Truncating to fit within model limits.")
        allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_parse, MODEL)
        user_prompt = truncate_to_token_limit(user_prompt, allowed_user_tokens, MODEL)

    # Step 1: Parse user input to extract components and flows
    parsed_json_str = call_wrapper_api(system_prompt_parse, user_prompt)
    if not parsed_json_str:
        print("Failed to parse the input.")
        exit(1)

    try:
        parsed_json = json.loads(parsed_json_str)
    except Exception as e:
        print("Failed to parse JSON from model output:", e)
        print(parsed_json_str)
        exit(1)

    # Step 2: Generate PlantUML code
    plantuml_code = call_wrapper_api(system_prompt_plantuml, json.dumps(parsed_json))
    if not plantuml_code:
        print("Failed to generate PlantUML code.")
        exit(1)

    print("\n*** Generated PlantUML Code ***\n")
    print(plantuml_code)

    # Step 3: Validate PlantUML code
    validation = call_wrapper_api(system_prompt_validate, plantuml_code)
    print("\n*** Validation ***\n")
    print(validation)

    # Step 4: If not valid, correct the code
    if validation.strip().lower() != "valid":
        corrected_code = call_wrapper_api(system_prompt_correct, plantuml_code)
        if corrected_code:
            print("\n*** Corrected PlantUML Code ***\n")
            print(corrected_code)
            plantuml_code = corrected_code
        else:
            print("Failed to correct the PlantUML code.")

    # Step 5: Render the PlantUML code to an image
    render_plantuml_locally(plantuml_code, "diagram.png", PLANTUML_JAR_PATH)

    print("\nThe diagram image is saved as diagram.png in current directory.")
