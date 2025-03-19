# text_data_processing.py

import re
import os
import time
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from data_processing_common import sanitize_filename
from gpt_utils import gpt_generate_text_single_prompt

def generate_text_metadata_single_prompt_gpt(input_text):
    """
    Uses GPT to generate:
    1) Summary (max 150 characters)
    2) Folder name (max 2 words)
    3) File name (max 3 words, underscores)
    """
    system_prompt = (
        "You are an assistant that helps generate summary, folder name, and file name from a document."
    )
    user_prompt = f"""
Please do the following for the given text:

1) A concise summary (up to 150 characters).
2) A folder name (max 2 words, nouns only).
3) A file name (max 3 words, nouns only, connected by underscores).

Text:
\"\"\"{input_text}\"\"\"

Output format (each on a new line):
[Summary]
[FolderName]
[FileName]
"""

    response = gpt_generate_text_single_prompt(system_prompt, user_prompt, max_tokens=400)
    lines = [line.strip() for line in response.split('\n') if line.strip()]

    summary = lines[0] if len(lines) > 0 else "No summary"
    folder_name = lines[1] if len(lines) > 1 else "documents"
    filename = lines[2] if len(lines) > 2 else "document"

    return summary, folder_name, filename

def process_single_text_file(args, silent=False, log_file=None):
    """
    Processes a single text file by calling GPT for summary/folder/file name.
    """
    file_path, text = args
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(file_path)}", total=1.0)

        # Call GPT
        summary, folder_name, filename_raw = generate_text_metadata_single_prompt_gpt(text)

        # Sanitize
        sanitized_foldername = sanitize_filename(folder_name, max_words=2)
        sanitized_filename = sanitize_filename(filename_raw, max_words=3)

        progress.update(task_id, advance=1.0)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {file_path}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"Description: {summary}\n"
        f"Folder name: {sanitized_foldername}\n"
        f"Generated filename: {sanitized_filename}\n"
    )
    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    else:
        print(message)

    return {
        'file_path': file_path,
        'foldername': sanitized_foldername,
        'filename': sanitized_filename,
        'description': summary
    }

def process_text_files(text_tuples, silent=False, log_file=None):
    """
    Processes text files sequentially, using GPT for each file.
    """
    results = []
    for args in text_tuples:
        data = process_single_text_file(args, silent=silent, log_file=log_file)
        results.append(data)
    return results
