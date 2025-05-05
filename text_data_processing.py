# text_data_processing.py

import os
import time
import openai
import tiktoken
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from data_processing_common import sanitize_filename
from gpt_utils import gpt_generate_text_single_prompt


##############################################################################
# 1. 짧은 문서 요약(기존 단일 Prompt)
##############################################################################
def generate_text_metadata_single_prompt_gpt(input_text):
    """
    Uses GPT to generate:
    1) Summary (max 150 characters)
    2) Category (one from the given 10 categories)
    3) Folder name (max 2 words)
    4) File name (max 3 words, underscores)

    The GPT response must be in the following format (line by line):
    [Summary]
    [Category]
    [FolderName]
    [FileName]
    """
    system_prompt = (
        "You are an AI assistant that analyzes a given text and provides a concise summary, "
        "a single category from the list, a short folder name, and a short file name."
    )
    user_prompt = f"""
Please do the following for the given text:

1) A concise summary (up to 150 characters).
2) A single classification chosen ONLY from these categories:
   [Technical, Academic, Business, Legal, News/Article, Creative, Personal, Instruction/Guide, Code/Programming, General Document]
3) A folder name (max 2 words, nouns only).
4) A file name (max 3 words, nouns only, connected by underscores).

Text:
\"\"\"{input_text}\"\"\"


Output format (each item on a new line, in this exact order):
[Summary]
[Category]
[FolderName]
[FileName]
"""

    # GPT 호출 (단일 프롬프트)
    response = gpt_generate_text_single_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=400  # 필요시 조정
    )

    # 응답 파싱
    lines = [line.strip() for line in response.split('\n') if line.strip()]

    summary = lines[0] if len(lines) > 0 else "No summary"
    category = lines[1] if len(lines) > 1 else "General Document"
    folder_name = lines[2] if len(lines) > 2 else "documents"
    filename = lines[3] if len(lines) > 3 else "document"

    return summary, category, folder_name, filename


##############################################################################
# 2. 긴 문서 요약(30,000 토큰 초과 시) - Map-Reduce Summarization
##############################################################################

def measure_token_length(text, model_name="gpt-4o"):
    """
    tiktoken으로 text를 인코딩해 실제 토큰 길이를 반환.
    gpt-4.1 기준 인코딩: tiktoken.encoding_for_model("gpt-4.1")
    """
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    return len(tokens)

def split_text_into_chunks(text, model_name="gpt-4o", chunk_token_limit=3000):
    """
    text를 chunk_token_limit 이하가 되도록 분할.
    30,000이 아니라, 안전하게 여유분 감안해서 (예: 3천 ~ 5천 단위) 나눌 수 있음.
    """
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    print("len(tokens): ", len(tokens))
    chunks = []
    start = 0
    while start < len(tokens):
        print("start: {}", format(start))
        end = start + chunk_token_limit
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

def gpt_summarize_text_chunk(text, system_prompt=None, user_prompt=None, max_tokens=400):
    """
    Map 단계: 한 chunk 요약 용. (간단 버전)
    """
    if not system_prompt:
        system_prompt = "You are a helpful assistant that summarizes text chunk by chunk."
    if not user_prompt:
        user_prompt = f"Please summarize this chunk:\n{text}"

    import openai
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )
    summary = response.choices[0].message.content.strip()
    return summary

def map_reduce_summarize(text, model_name="gpt-4.1", chunk_token_limit=3000):
    """
    긴 text -> 여러 chunk -> chunk별 요약(sub-summaries) -> 최종 통합 요약
    """
    MAX_CHUNKS = 7  # 최대 10개 chunk만 요약
    # 1) chunk 분할
    chunks = split_text_into_chunks(text, model_name=model_name, chunk_token_limit=chunk_token_limit)
    chunks = chunks[:MAX_CHUNKS]
    sub_summaries = []

    # 2) 각 chunk 요약 (Map)
    for i, chunk_text in enumerate(chunks):
        sub = gpt_summarize_text_chunk(
            text=chunk_text,
            user_prompt=f"Below is chunk #{i+1}:\n{chunk_text}\nSummarize it concisely."
        )
        sub_summaries.append(sub)
        time.sleep(0.3)  # rate-limit 방지용(필요시 조절)

    # 3) 부분 요약들 합쳐 최종 요약 (Reduce)
    combined = "\n\n".join(sub_summaries)
    final_summary = gpt_summarize_text_chunk(
        text=combined,
        system_prompt="You are an AI assistant that merges partial summaries into a final summary.",
        user_prompt=f"Combine these partial summaries into a cohesive overall summary:\n{combined}",
        max_tokens=800
    )

    return final_summary

def map_reduce_category_and_metadata(text):
    """
    최종 요약 얻은 뒤,
    - 카테고리 분류
    - 폴더명, 파일명 생성
    (단일 프롬프트로 4줄 결과 도출)
    """
    final_summary = map_reduce_summarize(text, model_name="gpt-4.1")

    # 이제 final_summary를 넣어, 기존 single-prompt와 같은 4줄(요약, 카테고리, 폴더, 파일명)을 생성
    system_prompt = (
        "You are an AI assistant that, given a summary, will produce a final summary (<=150 chars), "
        "a single category from the list, a folder name (2 words), and a file name (3 words underscores)."
    )
    user_prompt = f"""
We have an overall summary:
\"\"\"{final_summary}\"\"\"

Now please produce the following (line by line):
[ShortSummary <= 150 chars]
[Category: one from [Technical, Academic, Business, Legal, News/Article, Creative, Personal, Instruction/Guide, Code/Programming, General Document]]
[FolderName: 2 words, nouns]
[FileName: 3 words, nouns with underscores]
"""

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=400,
        temperature=0.5
    )
    lines = [line.strip() for line in response.choices[0].message.content.strip().split("\n") if line.strip()]

    short_summary = lines[0] if len(lines) > 0 else final_summary[:150]
    category = lines[1] if len(lines) > 1 else "General Document"
    folder_name = lines[2] if len(lines) > 2 else "documents"
    file_name = lines[3] if len(lines) > 3 else "document"

    return short_summary, category, folder_name, file_name

def generate_category_and_filename_single_prompt(text):
    """
    GPT를 이용해 category와 filename만 생성 (단일 프롬프트 방식)
    """
    system_prompt = (
        "You are an AI assistant that analyzes a given text and returns only:\n"
        "1) A single category from the list\n"
        "2) A descriptive file name (3 to 6 words, may include key nouns and adjectives, connected by underscores, clearly reflecting the document`s content:\n"
    )

    user_prompt = f"""
Given the following text:
\"\"\"{text}\"\"\"

Output the result in this format (each on a new line):
[Category: one of Technical, Academic, Business, Legal, News/Article, Creative, Personal, Instruction/Guide, Code/Programming, General Document]
[FileName: Generate a descriptive file name (3~6 words, nouns and adjectives allowed, underscore-separated, clearly reflects the document content)]
"""

    response = gpt_generate_text_single_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=200
    )

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    category = lines[0] if len(lines) > 0 else "General Document"
    filename = lines[1] if len(lines) > 1 else "default_filename"

    return category, filename

def map_reduce_category_and_filename(text):
    """
    긴 텍스트에 대해: map-reduce 요약 후 → category와 filename 생성
    """
    final_summary = map_reduce_summarize(text, model_name="gpt-4.1")

    system_prompt = (
        "You are an AI assistant that receives a summary and returns only:\n"
        "1) A single category from the list\n"
        "2) A file name (3 nouns with underscores)"
    )

    user_prompt = f"""
Given the summary:
\"\"\"{final_summary}\"\"\"

Output the following (each on a new line):
[Category: one of Technical, Academic, Business, Legal, News/Article, Creative, Personal, Instruction/Guide, Code/Programming, General Document]
[FileName: 3 words, nouns only, connected by underscores]
"""

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=200,
        temperature=0.5
    )

    lines = [line.strip() for line in response.choices[0].message.content.strip().split("\n") if line.strip()]
    category = lines[0] if len(lines) > 0 else "General Document"
    filename = lines[1] if len(lines) > 1 else "default_filename"

    return category, filename

##############################################################################
# 3. process_single_text_file: 길이 체크 후, 짧으면 single-prompt, 길면 map-reduce
##############################################################################

def process_single_text_file(args, ext, silent=False, log_file=None):
    file_path, text = args
    start_time = time.time()

    with (Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
    ) as progress):
        task_id = progress.add_task(f"Processing {os.path.basename(file_path)}", total=1.0)

        token_count = measure_token_length(text, model_name="gpt-4o")

        # (A) 엑셀/CSV 파일인가?
        if ext in [".xls", ".xlsx", ".csv"]:
            # 1) 토큰 검사. 30k 이하인지 여부
            if token_count <= 30000:
                # 짧은 경우 -> 기존 single spreadsheet logic
                category, filename = process_spreadsheet_file(file_path,columns=[0, 1, 2],model="gpt-4.1")
            else:
                # 긴 경우 -> map-reduce spreadsheet
                result = process_single_spreadsheet_mapreduce(file_path, silent=silent, log_file=log_file)

                # 끝나면 result를 그대로 return
                progress.update(task_id, advance=1.0)
                end_time = time.time()
                time_taken = end_time - start_time
                message = (
                    f"File: {file_path}\n"
                    f"Token Count: {token_count}\n"
                    f"Time taken: {time_taken:.2f} seconds\n"
                )
                if not silent:
                    print(message)

                return {
                    **result,
                    "token_count": token_count
                }

        else:
            # (B) 일반 텍스트
            if token_count <= 30000:
                category, filename = generate_category_and_filename_single_prompt(text)
            else:
                category, filename = map_reduce_category_and_filename(text)

        # 폴더명, 파일명 정제
        sanitized_foldername = sanitize_filename(category, max_words=2)
        sanitized_filename = sanitize_filename(filename, max_words=3)
        progress.update(task_id, advance=1.0)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {file_path}\n"
        f"Token Count: {token_count}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"Category: {category}\n"
        f"Folder name: {sanitized_foldername}\n"
        f"Generated filename: {sanitized_filename}\n"
    )
    print(message)
    if not silent:
        print(message)

    return {
        'file_path': file_path,
        'token_count': token_count,
        'foldername': sanitized_foldername,
        'filename': sanitized_filename,
        'category': category
    }

def map_reduce_summarize_spreadsheet_only(file_path, model_name="gpt-4.1", columns=[0, 1, 2], chunk_size=500):
    """
    큰 spreadsheet 파일에서 요약만 추출하는 함수 (카테고리, 파일명 생성은 별도로 진행).
    """
    import pandas as pd
    import math

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, usecols=columns)
    elif ext == ".csv":
        df = pd.read_csv(file_path, usecols=columns)
    else:
        raise ValueError(f"Unsupported spreadsheet extension: {ext}")

    total_rows = len(df)
    num_chunks = math.ceil(total_rows / chunk_size)

    sub_summaries = []

    for i in range(num_chunks):
        chunk_df = df[i*chunk_size:(i+1)*chunk_size]
        chunk_str = chunk_df.to_string(index=False)

        summary = gpt_summarize_text_chunk(
            text=chunk_str,
            user_prompt=f"Below is spreadsheet chunk #{i+1}:\n{chunk_str}\nSummarize it concisely.",
            model=model_name,
            max_tokens=500
        )
        sub_summaries.append(summary)
        time.sleep(0.3)  # Rate limit 보호용

    combined = "\n\n".join(sub_summaries)

    final_summary = gpt_summarize_text_chunk(
        text=combined,
        system_prompt="You are an AI assistant that merges spreadsheet summaries into a final summary.",
        user_prompt=f"Combine these partial spreadsheet summaries into a cohesive final summary:\n{combined}",
        model=model_name,
        max_tokens=800
    )

    return final_summary

def process_single_spreadsheet_mapreduce(file_path, silent=False, log_file=None):
    """
    Performs map-reduce summarization for a large spreadsheet,
    then generates category and filename based on that.
    """
    start_time = time.time()

    print(f"[MapReduce] Processing {os.path.basename(file_path)} ...")

    # (A) 요약만 먼저
    summary = map_reduce_summarize_spreadsheet_only(
        file_path,
        model_name="gpt-4.1",
        columns=[0, 1, 2],
        chunk_size=500
    )

    # (B) 요약 → 카테고리 및 파일명 생성
    category, file_name_raw = generate_category_and_filename_single_prompt(summary)
    sanitized_filename = sanitize_filename(file_name_raw, max_words=3)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {file_path}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"Category: {category}\n"
        f"Generated filename: {sanitized_filename}\n"
    )

    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + "\n")
    else:
        print(message)

    return {
        "file_path": file_path,
        "filename": sanitized_filename,
        "category": category
    }


def process_spreadsheet_file(file_path, columns=[0, 1, 2], model="gpt-3.5-turbo"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, usecols=columns)
    elif ext == ".csv":
        df = pd.read_csv(file_path, usecols=columns)
    else:
        raise ValueError(f"Unsupported spreadsheet extension: {ext}")

    # 데이터프레임 → 문자열
    df_str = df.to_string(index=False)
    category, filename_raw = generate_category_and_filename_single_prompt(df_str)
    sanitized_filename = sanitize_filename(filename_raw, max_words=3)

    return category, sanitized_filename

def process_single_spreadsheet(args, silent=False, log_file=None):
    """
    1) args = (file_path, None or placeholder)
    2) read columns 1,2,3 from the spreadsheet with pandas
    3) send to GPT for summarization
    4) return or store the result
    """
    file_path = args[0]
    start_time = time.time()

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(file_path)}", total=1.0)

        summary, category, folder_name_raw, filename_raw = process_spreadsheet_file(file_path, columns=[0, 1, 2], model="gpt-4.1")

        # 단순 예시로 folder/file 이름은 임의로
        # folder_name = "spreadsheet_summary"
        # file_name = "spreadsheet_file"
        sanitized_foldername = sanitize_filename(folder_name_raw, max_words=2)
        sanitized_filename = sanitize_filename(filename_raw, max_words=3)
        progress.update(task_id, advance=1.0)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {file_path}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"Summary:\n{summary}\n"
        f"Folder name: {sanitized_foldername}\n"
        f"Generated filename: {sanitized_filename}\n"
    )

    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + "\n")
    else:
        print(message)

    return {
        "file_path": file_path,
        "foldername": sanitized_foldername,
        "filename": sanitized_filename,
        "description": summary,
        "category": "Spreadsheet"
    }

def map_reduce_spreadsheet(file_path, model_name="gpt-4.1", columns=[0,1,2], chunk_size=500):
    """
    1) Excel/CSV 로딩
    2) (행 수가 많다면) chunk_size씩 나누어 부분 요약 (Map)
    3) 부분 요약들을 다시 GPT로 합쳐 최종 요약(Reduce)
    4) 최종 요약을 다시 '메타데이터(4줄: Summary, Category, Folder, FileName)'로 변환
    """
    import pandas as pd
    import openai
    import tiktoken
    import time

    # A. 스프레드시트 로딩
    import os
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, usecols=columns)
    elif ext == ".csv":
        df = pd.read_csv(file_path, usecols=columns)
    else:
        raise ValueError(f"Unsupported spreadsheet extension: {ext}")

    # B. 데이터프레임을 여러 조각(chunks)으로 분할
    #    여기서는 행 기준 chunk_size씩 쪼갠다.
    total_rows = len(df)
    chunks = []
    start_idx = 0
    while start_idx < total_rows:
        end_idx = min(start_idx + chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]
        chunk_str = df_chunk.to_string(index=False)
        chunks.append(chunk_str)
        start_idx = end_idx

    # C. Map 단계: 각 chunk를 GPT로 요약
    def summarize_chunk(chunk_text):
        system_prompt = "You are an AI assistant summarizing partial spreadsheet data."
        user_prompt = f"Below is partial table data:\n{chunk_text}\nSummarize the main points."
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user",   "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    sub_summaries = []
    for i, ctext in enumerate(chunks):
        sub_summ = summarize_chunk(ctext)
        sub_summaries.append(sub_summ)
        time.sleep(0.3)  # rate-limit 대비

    # D. Reduce 단계: 부분 요약들을 하나로 합쳐 최종 요약
    combined_subs = "\n\n".join(sub_summaries)
    reduce_system_prompt = "You are an AI assistant that merges partial summaries into a final summary."
    reduce_user_prompt = f"Combine these partial summaries of spreadsheet data into one cohesive summary:\n{combined_subs}"
    reduce_response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": reduce_system_prompt},
            {"role": "user",   "content": reduce_user_prompt}
        ],
        max_tokens=800,
        temperature=0.5
    )
    final_summary_text = reduce_response.choices[0].message.content.strip()

    # E. 최종 요약을 "4줄 형식"으로 변환 (Summary, Category, Folder, FileName)
    meta_system_prompt = (
        "You are an AI assistant that, given a summary of spreadsheet data, produces "
        "1) a short summary(<=150 chars), "
        "2) one category from [Technical, Academic, Business, Legal, News/Article, Creative, Personal, Instruction/Guide, Code/Programming, General Document], "
        "3) a folder name(2 words, nouns), "
        "4) a file name(3 words, underscores)."
    )
    meta_user_prompt = f"""
We have a final summary of the spreadsheet:
\"\"\"{final_summary_text}\"\"\"

Output format (4 lines):
[ShortSummary]
[Category]
[FolderName]
[FileName]
"""
    meta_resp = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role":"system","content":meta_system_prompt},
            {"role":"user","content":meta_user_prompt}
        ],
        max_tokens=400,
        temperature=0.5
    )
    lines = [line.strip() for line in meta_resp.choices[0].message.content.strip().split("\n") if line.strip()]

    short_summary = lines[0] if len(lines) > 0 else final_summary_text[:150]
    category = lines[1] if len(lines) > 1 else "General Document"
    folder_name = lines[2] if len(lines) > 2 else "documents"
    file_name = lines[3] if len(lines) > 3 else "document"

    return short_summary, category, folder_name, file_name


def process_text_files(text_tuples, file_list, silent=False, log_file=None):
    """
    Processes text files sequentially.
    If short enough, single prompt GPT. If too long, map-reduce approach.
    """
    data_list = []
    for file_path, text in text_tuples:
        original_info = next((f for f in file_list if f['file_path'] == file_path), {})
        ext = os.path.splitext(file_path)[1]
        data = process_single_text_file((file_path, text), ext, silent=silent, log_file=log_file)
        data.update({
            "fileId": original_info.get("fileId"),
            "name": original_info.get("name"),
            "fileType": original_info.get("fileType"),
            "size": original_info.get("size"),
            "createdAt": original_info.get("createdAt"),
        })
        data_list.append(data)
    return data_list
