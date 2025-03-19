# image_data_processing.py

import os
import time
import base64
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# 사용자 예시 코드 기반 (from openai import OpenAI)
from openai import OpenAI

def encode_image_to_data_url(image_path):
    """
    Reads the local image file, base64-encodes it,
    and returns a data URL (e.g. 'data:image/jpeg;base64,<encoded>').
    For simplicity, we assume jpeg.
    If you have png, you may want to set 'image/png' etc. accordingly.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    # 아래는 JPEG이라고 가정
    data_url = f"data:image/jpeg;base64,{encoded}"
    return data_url

def analyze_image_with_gpt_data_url(data_url):
    """
    Sends a data URL image to GPT for analysis using the user snippet approach.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",  # 사용자 예시: gpt-4o
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
    )
    # GPT 응답 텍스트
    return completion.choices[0].message.content

def generate_folder_and_filename_from_gpt_description(description):
    """
    GPT에서 받은 텍스트(이미지 설명)를 기반으로,
    간단 규칙으로 폴더명, 파일명을 만들어내는 예시 로직.
    """
    import re
    words = re.findall(r"[a-zA-Z]+", description)
    folder_words = words[:2] if words else ["images"]
    file_words = words[:3] if words else ["image"]

    foldername = "_".join(word.lower() for word in folder_words)
    filename = "_".join(word.lower() for word in file_words)

    if not foldername:
        foldername = "images"
    if not filename:
        filename = "image"
    return foldername, filename

def process_single_image(image_path, silent=False, log_file=None):
    """
    1) 로컬 이미지 파일을 data URL로 변환 (base64).
    2) GPT에 전송하여 이미지 설명 획득.
    3) 설명을 바탕으로 폴더/파일명 생성.
    4) 결과(dict) 반환.
    """
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(image_path)}", total=1.0)

        # 1) base64로 인코딩 후 data URL 만들기
        data_url = encode_image_to_data_url(image_path)

        # 2) GPT 분석
        gpt_description = analyze_image_with_gpt_data_url(data_url)

        # 3) 폴더명/파일명 생성
        foldername, filename = generate_folder_and_filename_from_gpt_description(gpt_description)

        progress.update(task_id, advance=1.0)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {image_path}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"GPT Description: {gpt_description}\n"
        f"Folder name: {foldername}\n"
        f"Generated filename: {filename}\n"
    )
    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    else:
        print(message)

    return {
        "file_path": image_path,
        "foldername": foldername,
        "filename": filename,
        "description": gpt_description
    }

def process_image_files(image_paths, silent=False, log_file=None):
    """
    (A) image_paths 안에 있는 로컬 이미지 파일들을
    순차적으로 GPT에 전달하여 분석하는 함수.
    (B) base64 인코딩 + data URL를 생성 -> GPT 전송
    """
    data_list = []
    for image_path in image_paths:
        # 이미지 처리
        data = process_single_image(
            image_path=image_path,
            silent=silent,
            log_file=log_file
        )
        data_list.append(data)
    return data_list
