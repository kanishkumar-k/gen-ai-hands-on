import json
import requests
import tiktoken
import ast
import time
import sys
import os

API_KEY=""
API_URL=""
MODEL = "gpt-4"
MAX_MODEL_TOKENS = 4096
RESPONSE_TOKEN_BUFFER = 1200
MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - RESPONSE_TOKEN_BUFFER

BEST_PRACTICES_CHECKLIST = """
Best-Practices Docstring Checklist:
- Clearly describes the function's purpose.
- Lists all parameters and their types.
- Describes the return value and its type.
- Mentions exceptions raised, if any.
- Uses proper formatting (e.g., Google, NumPy, or Sphinx style).
- Is concise and user-focused.
"""

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
        "temperature": 0.2,
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

def extract_functions_and_docstrings(py_file_path):
    with open(py_file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            docstring = ast.get_docstring(node)
            functions.append({"name": name, "docstring": docstring})
    return functions

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_api_docs.py <python_file>")
        sys.exit(1)

    input_py_file = sys.argv[1]
    if not os.path.exists(input_py_file):
        print(f"File not found: {input_py_file}")
        sys.exit(1)

    output_md_file = "API_DOCUMENTATION.md"
    functions = extract_functions_and_docstrings(input_py_file)
    documentation_entries = []

    for func in functions:
        name = func["name"]
        docstring = func["docstring"]

        if not docstring:
            print(f"No docstring found for function '{name}'. Generating docstring first...")
            with open(input_py_file, "r", encoding="utf-8") as f:
                source_lines = f.readlines()

            func_code = ""
            in_func = False
            indent_level = None
            for line in source_lines:
                if line.strip().startswith(f"def {name}("):
                    in_func = True
                    indent_level = len(line) - len(line.lstrip())
                if in_func:
                    func_code += line
                    if line.strip() == "" or (len(line) - len(line.lstrip()) <= indent_level and line.strip()):
                        break

            # Generate docstring from the function code
            system_prompt_docstring = (
                "You are an expert Python developer. Write a high-quality docstring for the following function."
            )
            user_prompt_docstring = f"Function code:\n{func_code}\n"
            docstring = call_wrapper_api(system_prompt_docstring, user_prompt_docstring)
            print(f"Docstring for function {name}:\n{docstring}\n")
            if not docstring:
                docstring = "*Failed to generate docstring.*"

        # Step 1: Transform docstring to user-facing documentation
        system_prompt_transform = (
            "You are an expert technical writer. Transform the following Python function/code docstring "
            "into a clear, user-facing API documentation entry. Use Markdown formatting."
        )
        user_prompt_transform = f"Function: {name}\nDocstring:\n{docstring}\n"

        total_tokens = count_tokens(system_prompt_transform, MODEL) + count_tokens(user_prompt_transform, MODEL)
        if total_tokens > MAX_INPUT_TOKENS:
            print(f"Input for function '{name}' is too long ({total_tokens} tokens). Truncating docstring.")
            allowed_user_tokens = MAX_INPUT_TOKENS - count_tokens(system_prompt_transform, MODEL)
            user_prompt_transform = f"Function: {name}\nDocstring:\n{truncate_to_token_limit(docstring, allowed_user_tokens, MODEL)}\n"

        user_friendly_doc = call_wrapper_api(system_prompt_transform, user_prompt_transform)
        if not user_friendly_doc:
            user_friendly_doc = "*Failed to generate documentation.*"

        # Step 2: Check against best-practices checklist
        system_prompt_check = (
            "You are a Python documentation reviewer. "
            "Check the following docstring against the best-practices checklist and suggest improvements if needed."
        )
        user_prompt_check = (
            f"Function: {name}\n"
            f"Docstring:\n{docstring}\n\n"
            f"{BEST_PRACTICES_CHECKLIST}\n"
            "If the docstring is already excellent, say so. Otherwise, provide a revised version."
        )

        total_tokens_check = count_tokens(system_prompt_check, MODEL) + count_tokens(user_prompt_check, MODEL)
        if total_tokens_check > MAX_INPUT_TOKENS:
            print(f"Best-practices check input for function '{name}' is too long ({total_tokens_check} tokens). Truncating docstring.")
            allowed_user_tokens_check = MAX_INPUT_TOKENS - count_tokens(system_prompt_check, MODEL) - count_tokens(BEST_PRACTICES_CHECKLIST, MODEL)
            user_prompt_check = (
                f"Function: {name}\n"
                f"Docstring:\n{truncate_to_token_limit(docstring, allowed_user_tokens_check, MODEL)}\n\n"
                f"{BEST_PRACTICES_CHECKLIST}\n"
                "If the docstring is already excellent, say so. Otherwise, provide a revised version."
            )

        best_practices_feedback = call_wrapper_api(system_prompt_check, user_prompt_check)
        if not best_practices_feedback:
            best_practices_feedback = "*Failed to review docstring.*"

        entry = f"## `{name}`\n\n{user_friendly_doc}\n\n**Best-Practices Review:**\n{best_practices_feedback}\n"
        documentation_entries.append(entry)

    with open(output_md_file, "w", encoding="utf-8") as f:
        f.write("# API Documentation\n\n")
        for entry in documentation_entries:
            f.write(entry + "\n---\n")
    print(f"Documentation written to {output_md_file}")

if __name__ == "__main__":

    main()
