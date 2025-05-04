# image_data_processing.py

import os
import time
import base64
import io
import re
import nltk
from dotenv import load_dotenv
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from openai import OpenAI
from PIL import Image
from data_processing_common import sanitize_filename
#from gensim.summarization import keywords

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

##############################################################################
# 1) 메모리에서 이미지 리사이즈 (BytesIO)
##############################################################################
def resize_image_in_memory(input_path, max_size, quality):
    """
    Loads the image from 'input_path' in memory,
    resizes it so that neither dimension exceeds 'max_size',
    and saves as JPEG (quality=70) to an in-memory buffer (BytesIO).
    Returns the raw bytes of the resized JPEG.
    """
    with Image.open(input_path) as img:
        # 변환 (예: RGB)
        img = img.convert("RGB")
        # 최대 크기 제한
        img.thumbnail((max_size, max_size))
        # 메모리에 저장
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return buffer.read()  # 바이트로 반환


##############################################################################
# 2) 로컬 이미지 파일 -> In-memory로 리사이즈 -> Base64 data URL
##############################################################################
def encode_image_to_data_url_in_memory(input_path, max_size, quality):
    """
    Performs an in-memory resize of the image, then base64-encodes it,
    returning a data URL (e.g. 'data:image/jpeg;base64,<encoded>').
    """
    resized_bytes = resize_image_in_memory(input_path, max_size, quality)
    encoded = base64.b64encode(resized_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"
    return data_url

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


##############################################################################
# 3) GPT 호출 및 이름 생성: 단일 summary를 기반으로 폴더명, 파일명을 생성
##############################################################################

def generate_folder_and_filename_from_summary(summary):
    """
    주어진 summary 텍스트에서 폴더명과 파일명을 생성합니다.
    - 폴더명: 불용어(stopwords)를 제거한 후 처음 2개의 단어 (소문자, 언더스코어 연결)
    - 파일명: 불용어를 제거한 후 처음 3개의 단어 (소문자, 언더스코어 연결)
    충분한 단어가 없으면 기본값("images", "image")을 사용합니다.
    """
    # 알파벳 단어만 추출
    words = re.findall(r"[a-zA-Z]+", summary)
    # 불용어 목록 (일반적인 관사용 단어들을 제거)
    stopwords = {"the", "a", "an", "this", "image", "shows", "depicts",
                 "is", "are", "of", "and", "to", "in", "as", "with", "that", "on", "at", "by"}
    filtered_words = [word.lower() for word in words if word.lower() not in stopwords]

    # 만약 충분한 정보가 없으면 기본값 지정
    if not filtered_words:
        filtered_words = ["unknown"]

    folder_words = filtered_words[:2] if len(filtered_words) >= 2 else filtered_words
    file_words = filtered_words[:3] if len(filtered_words) >= 3 else filtered_words

    folder_name = "_".join(folder_words) if folder_words else "images"
    file_name = "_".join(file_words) if file_words else "image"

    return folder_name, file_name

def generate_names_with_pos(summary):
    """
    summary에서 단어를 토큰화한 후, 명사(NN, NNS, NNP, NNPS)만 추출하여
    앞 2개를 폴더명, 앞 3개를 파일명으로 사용합니다.
    """
    tokens = nltk.word_tokenize(summary)
    tagged = nltk.pos_tag(tokens)
    # 명사만 필터링
    nouns = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    print("nouns: ", nouns)
    if not nouns:
        nouns = ["unknown"]
    folder_words = nouns[:2] if len(nouns) >= 2 else nouns
    file_words = nouns[:3] if len(nouns) >= 3 else nouns
    folder_name = "_".join(word.lower() for word in folder_words)
    file_name = "_".join(word.lower() for word in file_words)
    print("folder_name: ", folder_name)
    print("file_name: ", file_name)
    return folder_name, file_name

def generate_names_with_gpt(summary, api_key):
    """
    GPT를 호출하여 summary를 기반으로 폴더명과 파일명을 생성합니다.
    """
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an AI assistant that creates concise folder and file names based on a summary."
    )
    user_prompt = f"""
Based on the summary below, generate:
1) A folder name that reflects the main subject (max 2 words, nouns only).
2) A file name that captures the essence of the image (max 3 words, nouns only, connected by underscores).

If the summary is unclear, output 'unclear' for both.

Summary:
"{summary}"

Output format (each item on its own line):
[FolderName]
[FileName]
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=200,
        temperature=0.7,
    )
    response_text = completion.choices[0].message.content.strip()
    print("response_text: ", response_text)
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    print("lines: ", lines)
    folder_name = lines[0] if len(lines) > 0 else "images"
    file_name = lines[1] if len(lines) > 1 else "image"
    return folder_name, file_name
"""
def generate_names_with_textrank(summary):
    gensim의 keywords 함수를 사용하여 summary에서 키워드를 추출한 후,
    상위 2개 키워드를 폴더명, 상위 3개 키워드를 파일명으로 사용합니다.
    key_str = keywords(summary, words=5, lemmatize=True)
    key_list = [word for word in key_str.split('\n') if word.strip()]
    print("key_list: ", key_list)
    if not key_list:
        key_list = ["unknown"]
    folder_words = key_list[:2] if len(key_list) >= 2 else key_list
    print("folder_words: ", folder_words)
    file_words = key_list[:3] if len(key_list) >= 3 else key_list
    print("file_words: ", file_words)
    folder_name = "_".join(word.lower() for word in folder_words)
    file_name = "_".join(word.lower() for word in file_words)
    return folder_name, file_name
"""
# 관용사 및
def generate_names_with_extended_rules(summary):
    """
    summary에서 알파벳 단어를 추출한 후, 기본 불용어에 추가적으로
    "also", "features", "information", "display", "present" 등을 제거합니다.
    남은 단어 중 앞 2개를 폴더명, 3개를 파일명으로 사용합니다.
    """
    stopwords = {"the", "a", "an", "this", "image", "shows", "depicts",
                 "is", "are", "of", "and", "to", "in", "as", "with", "that", "on", "at", "by",
                 "also", "features", "information", "display", "present"}
    words = re.findall(r"[a-zA-Z]+", summary)
    filtered = [word.lower() for word in words if word.lower() not in stopwords]
    if not filtered:
        filtered = ["unknown"]
    print("filtered: ", filtered)
    return " ".join(filtered)

##############################################################################
# 3) GPT 호출: 한 번의 프롬프트 -> [ShortSummary], [FolderName], [FileName] (3줄)
##############################################################################
def generate_image_metadata_single_prompt(image_path, api_key=None):
    """
    1) In-memory resize the original image (no disk temp file).
    2) Base64-encode that resized image.
    3) Send a single GPT prompt that demands a strict 3-line output:
       [ShortSummary]
       [FolderName]
       [FileName]
    Returns (summary, folder_name, file_name).
    """

    # A) 메모리 리사이즈 + base64 인코딩
    #data_url = encode_image_to_data_url_in_memory(input_path=image_path, max_size=1024, quality=90)
    # data_url = "https://ifh.cc/g/ntQPT0.jpg"
    data_url = encode_image(image_path)
    # B) OpenAI Client
    client = OpenAI(api_key=api_key)

    # C) Prompt
    system_prompt = (
        "You are an AI assistant with advanced vision capabilities."
        "Analyze the image based solely on the visible information, and describe the clear elements you observe."
        "Do not invent details beyond what is clearly visible."
    )

    user_prompt = f"""
    Please do the following for the image data below:

    1) Provide a concise description of the main subject and its context (max ~150 characters). 
        - Describe only the clear elements; if certain details are ambiguous, simply omit them.
    2) Based on that description, generate a folder name (max 2 words, nouns only).
    3) Generate a file name (max 3 words, nouns only, connected by underscores).

    Image data (base64):
    {data_url}
    
Output format (each item on its own line):
[ShortSummary]
[FolderName]
[FileName]
"""

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = f"""
Please do the following for this image:

1) A short summary (1~2 sentences). Do not invent details not visible.
2) One category only from: [animal, nature, food, technology, abstract, architecture, vehicle, document, art, people, symbol, object, landscape, history].
3) A 2-word folder name: 
   - First word = chosen category
   - Second word = a noun describing the main subject
4) A 3-word file name (all nouns, connected by underscores), representing the essence of the image.

Output (each on its own line):
[ShortSummary]
[Category]
[FolderName]
[FileName]

Example:
--------------------------------------------------
A brief description of the image content.
animal
animal_whales
ocean_whale_breach
--------------------------------------------------

Image data:
{data_url}

"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant with advanced vision capabilities."
                    "When describing the image, you must also mention a broader or higher-level category that best fits the main subject."
                    "Choose the category from the following set only: animal, nature, food, technology, abstract, architecture, vehicle, document, art, people, symbol, object, landscape, history."
                    "Do not invent details beyond what is clearly visible."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Please describe this image in a short summary. Also provide a broad category. Then create a folder name (2 words) and a file name (3 words) based on that summary."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{data_url}",
                        },
                    },
                ],
            }
        ],
        max_tokens=600,
        temperature=0.7,
    )

    # D) 응답 파싱
    response_text = completion.choices[0].message.content.strip()
    print(response_text)
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    print(lines)
    short_summary = lines[0] if len(lines) > 0 else "No summary"
    category = lines[1] if len(lines) > 1 else "misc"
    raw_folder = lines[2] if len(lines) > 2 else "images"
    raw_file = lines[3] if len(lines) > 3 else "image"
    # 1. 기본
    #folder_name, file_name = generate_folder_and_filename_from_summary(response_text)
    # 2. nltk 사용
    #folder_name, file_name = generate_names_with_pos(generate_names_with_extended_rules(response_text))
    #return response_text, folder_name, file_name
    return short_summary, category, raw_file

##############################################################################
# 4) process_single_image: 3줄 출력 로직
##############################################################################
def process_single_image(image_path, api_key=None, silent=False, log_file=None):
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(image_path)}", total=1.0)

        # GPT로 이미지 요약+폴더+파일명 한 번에 받기 (인메모리 리사이즈)
        summary_raw, folder_raw, filename_raw = generate_image_metadata_single_prompt(
            image_path=image_path,
            api_key=api_key
        )

        # 정제
        sanitized_folder = sanitize_filename(folder_raw, max_words=2)
        sanitized_file = sanitize_filename(filename_raw, max_words=3)

        progress.update(task_id, advance=1.0)

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {image_path}\n"
        f"Time taken: {time_taken:.2f} seconds\n"
        f"Short Summary: {summary_raw}\n"
        f"Folder name: {folder_raw}\n"
        f"Generated filename: {sanitized_file}\n"
    )
    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + "\n")
    else:
        print(message)

    return {
        "file_path": image_path,
        "foldername": folder_raw,
        "filename": sanitized_file,
        "description": summary_raw
    }


##############################################################################
# 5) process_image_files: 여러 이미지 처리
##############################################################################
def process_image_files(image_paths, file_list, api_key=None, silent=False, log_file=None):
    data_list = []
    for image in image_paths:
        # 기존 metadata 참고용
        original_info = next((f for f in file_list if f['file_path'] == image), {})
        data = process_single_image(image, api_key=api_key, silent=silent, log_file=log_file)
        data.update({
            "fileId": original_info.get("fileId"),
            "name": original_info.get("name"),
            "fileType": original_info.get("fileType"),
            "size": original_info.get("size"),
            "createdAt": original_info.get("createdAt"),
        })
        data_list.append(data)
    return data_list
