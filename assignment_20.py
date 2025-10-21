import pandas as pd
import json
import requests
import time
import os
import tiktoken

API_KEY=""
API_URL=""
MODEL = "gpt-4"
RESPONSE_TOKEN_BUFFER = 800

def num_tokens_from_messages(messages, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4 
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  
    return num_tokens

def call_wrapper_api(system_prompt: str, user_prompt: str, rate_limit: int = 3) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    token_count = num_tokens_from_messages(messages, MODEL)
    if token_count > 8192 - RESPONSE_TOKEN_BUFFER:
        print(f"Prompt too long ({token_count} tokens). Please shorten your input.")
        return None

    payload = json.dumps({
        "messages": messages,
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
            print(f"Retrying ({trials + 1}/{rate_limit})")
            time.sleep(2)
    return None

def preprocess_stacked_format(input_file):
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return None
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading '{input_file}': {e}")
        return None
    if df.empty:
        print(f"Input file '{input_file}' is empty.")
        return None
    df[['Name', 'College Name', 'Roll No', 'School Number']] = df[['Name', 'College Name', 'Roll No', 'School Number']].ffill()
    df = df[df['Subject'].notna()]
    if df.empty:
        print("No subject data found after filtering.")
        return None
    pivot = df.pivot_table(index=['Name', 'College Name', 'Roll No', 'School Number'],
                           columns='Subject', values='Marks').reset_index()
    pivot.columns.name = None
    return pivot

def assign_departments(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mech = df[(df['Physics'] > 90) & (df['Chemistry'] > 85) & (df['Maths'] > 60)]
    comp = df[(df['Physics'] > 60) & (df['Chemistry'] > 60) & (df['Maths'] > 90)]
    elec = df[(df['Physics'] > 95) & (df['Chemistry'] > 60) & (df['Maths'] > 90)]
    return mech, comp, elec

def save_to_excel(df, filename):
    if df is not None and not df.empty:
        df.to_excel(filename, index=False)

def generate_report_summary(mech, comp, elec, total_students):
    summary = (
        f"Total students processed: {total_students}\n"
        f"Mechanical Engineering: {len(mech)} eligible students\n"
        f"Computer Engineering: {len(comp)} eligible students\n"
        f"Electrical Engineering: {len(elec)} eligible students\n\n"
        "Mechanical Engineering Criteria: Physics > 90%, Chemistry > 85%, Maths > 60%\n"
        "Computer Engineering Criteria: Physics > 60%, Chemistry > 60%, Maths > 90%\n"
        "Electrical Engineering Criteria: Physics > 95%, Chemistry > 60%, Maths > 90%\n"
    )
    return summary

def main():
    input_file = "InputData_Assignment20.xlsx"
    df = preprocess_stacked_format(input_file)
    if df is None or df.empty:
        print("No data to process. Exiting.")
        return

    mech, comp, elec = assign_departments(df)
    save_to_excel(mech, "Output/Mechanical_Engineering.xlsx")
    save_to_excel(comp, "Output/Computer_Engineering.xlsx")
    save_to_excel(elec, "Output/Electrical_Engineering.xlsx")

    summary = generate_report_summary(mech, comp, elec, len(df))
    print("\n--- Assignment Summary ---\n")
    print(summary)

    system_prompt = (
        "You are an expert academic counselor. Based on the following summary of department assignments, "
        "provide insights on the distribution, possible reasons for the trends, and any recommendations for students or educators."
    )
    ai_report = call_wrapper_api(system_prompt, summary)
  
    if ai_report:
        print("\n--- AI-Generated Report ---\n")
        print(ai_report)
        with open("Output/Assignment_Report.txt", "w", encoding="utf-8") as f:
            f.write("--- Assignment Summary ---\n")
            f.write(summary)
            f.write("\n--- AI-Generated Report ---\n")
            f.write(ai_report)
    else:
        print("Failed to generate AI report.")

if __name__ == "__main__":
    main()

