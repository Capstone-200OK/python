import os
from dateutil import parser
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def classify_filename_with_gpt(file_name):
    prompt = f"""
You are a file classification assistant.
Classify the file based only on its filename into one of the following categories:
Resume, Contract, Image, Source Code, Report, Financial Document, Meeting Notes, Presentation, Other.

Return only the best-fitting category.

Filename: "{file_name}"
Category:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT classification failed for '{file_name}': {e}")
        return "Other"

def process_files_by_title(file_list, output_path):
    operations = []
    for file_info in file_list:
        file_path = file_info.get("file_path")
        file_name = file_info.get("name")

        category = classify_filename_with_gpt(file_name)

        dir_path = "/".join([output_path, category])
        new_file_path = "/".join([dir_path, file_name])

        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': 'hardlink',
            'fileId': file_info.get("fileId"),
            'name': file_name,
            'fileType': file_info.get("fileType"),
            'size': file_info.get("size"),
            'createdAt': file_info.get("createdAt"),
        }
        operations.append(operation)
    return operations
