import pandas as pd
import json
import requests
import time
import os
import tiktoken

API_URL = "https://openai-api-wrapper-urtjok3rza-wl.a.run.app/api/chat/completions/"
API_KEY = "eyJraWQiOiI1cHpvN3Z0NnBiKzB3OEtwS3kwN0VWUzgzYnozSTZuVmhVT1A3dTlpYnlzPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoiQkJuNDBjbDlVZ1hMdHJJc0lTYk92ZyIsInN1YiI6IjJiYmY0YTVkLWFiMjMtNGM3My04ZDlkLWFhZjkyMTY0YjExNSIsImNvZ25pdG86Z3JvdXBzIjpbInVzLWVhc3QtMl9vSTk5c0RKT1RfRGVsb2l0dGUiXSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMi5hbWF6b25hd3MuY29tXC91cy1lYXN0LTJfb0k5OXNESk9UIiwiY3VzdG9tOm9yZ19pZCI6IkRlbG9pdHRlIiwiY3VzdG9tOmpvYl90aXRsZSI6IlNvZnR3YXJlIEVuZ2luZWVyIEkiLCJpZGVudGl0aWVzIjpbeyJ1c2VySWQiOiJOTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicHJvdmlkZXJOYW1lIjoiRGVsb2l0dGUiLCJwcm92aWRlclR5cGUiOiJPSURDIiwiaXNzdWVyIjpudWxsLCJwcmltYXJ5IjoidHJ1ZSIsImRhdGVDcmVhdGVkIjoiMTc0NDEwNTQ2MjY2OSJ9XSwiYXV0aF90aW1lIjoxNzU0Mzc0Nzk5LCJjdXN0b206ZW1haWxzIjoie1wicHJpbWFyeV9lbWFpbFwiOiBcImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb21cIiwgXCJzZWNvbmRhcnlfZW1haWxzXCI6IFtdfSIsImV4cCI6MTc1NDQ3MDM3OCwiaWF0IjoxNzU0NDY2Nzc5LCJqdGkiOiIxZWQ2YzYxZi1jNmUyLTQwZGYtOWM2MC02NTZmMjUxNjhkNzMiLCJlbWFpbCI6ImtrYXJ1bmFrYXJhbkBkZWxvaXR0ZS5jb20iLCJvcmdhbmlzYXRpb25fZGV0YWlscyI6Ilt7XCJuYW1lXCI6IFwiSGFzaGVkSW5cIiwgXCJ0ZW5hbnRfaWRcIjogXCIxMTIyXCIsIFwiYWxsb3dlZF9hcHBzXCI6IFtcImFsbG9jYXRpb25cIiwgXCJsZWF2ZXNcIiwgXCJyZXdhcmRzXCIsIFwiaHUtYXV0b21hdGlvblwiLCBcInRpbWVzaGVldFwiLCBcInB1bHNlXCIsIFwidm9sdW50ZWVyXCIsIFwiaG9tZVwiLCBcIm9wc01ldHJpY3NcIiwgXCJodS1ldmFsdWF0aW9uXCIsIFwiYXV0b2RpZGFjdFwiLCBcImt1ZG9zXCIsIFwicG9kc1wvcHVyc3VpdHNcIiwgXCJoaXJlXCIsIFwicG9kc1wiLCBcImN1bXVsdXNcIiwgXCJyZWxheVwiLCBcInJ0djNcIiwgXCJsZWFybmluZ25kZXZcIiwgXCJjYW1cIl0sIFwidGltZXpvbmVcIjogXCJHTVRcIiwgXCJjdXJyZW5jeVwiOiBcIklOUlwiLCBcImxvZ29fczNfb2JqZWN0X3VybFwiOiBcImh0dHBzOlwvXC9kbmEtcHJvZC1yZXNvdXJjZS5zMy51cy1lYXN0LTIuYW1hem9uYXdzLmNvbVwvb3JnX2RpcmVjdG9yeVwvb3JnYW5pc2F0aW9uX2xvZ29zXC9oYXNoZWRpbi5wbmdcIn1dIiwiY3VzdG9tOnV1aWQiOiI4NDVmYTI2MC00NDA0LTQxYjktYTIzNS0xYWFlNmMyNGE1NDIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImNvZ25pdG86dXNlcm5hbWUiOiJEZWxvaXR0ZV9OTTFfNGdmOHhmUGJvNk1wcnpqbXZhM1NPSHVHd0hJcXNlb0FHckNkaHZBIiwicGljdHVyZSI6Imh0dHBzOlwvXC9ncmFwaC5taWNyb3NvZnQuY29tXC92MS4wXC9tZVwvcGhvdG9cLyR2YWx1ZSIsIm9yaWdpbl9qdGkiOiI5ZDhlMzU1NS1iOTBjLTRlN2QtYTYwMS1jZjI3YzY5MzE3YjQiLCJhdWQiOiI1YnB0a3JzcnNtOWZkcHU5ODB2a2RoZ29vcCIsInRva2VuX3VzZSI6ImlkIiwibmFtZSI6IkthcnVuYWthcmFuLCBLYW5pc2hrdW1hciIsInByaW1hcnlfdGVuYW50IjoiMTEyMiIsInNlc3Npb25fdGVuYW50IjoiMTEyMiJ9.l3BAy4scQ7_o3i_8-1EmoVm896E77po4Ngnyni_SUFZxn--d9t1-WABCHIp0xceHlzRfdUCBGuzaOL8dV9HI1iWreNnR0O3_CV17-X4m1Zok_KWBoipI-9XlV5asWx-bVkVkoHCvt6oj0yCVfMRuhswzOIb8VdLFeO86xHqqoXjr-6bm-G4B7V_3KSw_XjrKA6rQF5Aavy5tkUbEG4EBQylY0USz_tWHzKrZ5I7qgMVVOny9yGfitwjOKQifYMaRlEAOt849oG5aXQXOzAi771SGOh1_Cm3hADBoh5MrvtZKEtwVgXyjGt9HRlsGFz0giKGucNubed3vfZ1YCKdIMg"
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
