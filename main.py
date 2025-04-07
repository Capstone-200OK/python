import os
import time
import nltk
import requests

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_file_data
)
from data_processing_common import (
    compute_operations,
    execute_operations,
    process_files_by_date,
    process_files_by_type
)
# GPT-based text processing (unchanged)
from text_data_processing import (
    process_text_files
)
# Image processing that requires image_url_map
from image_data_processing import (
    process_image_files
)

nltk.download('averaged_perceptron_tagger_eng')

BASE_URL = "http://localhost:8080"

def ensure_nltk_data():
    """
    Make sure NLTK data (stopwords, punkt, etc.) is installed.
    """
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ('yes', 'y'):
            return True
        elif response in ('no', 'n'):
            return False
        elif response == '/exit':
            print("Exiting program.")
            exit()
        else:
            print("Please enter 'yes' or 'no'. To exit, type '/exit'.")

def get_mode_selection():
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content (GPT-based text, naive images)")
        print("2. By Date")
        print("3. By Type")
        response = input("Enter 1, 2, or 3 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return 'content'
        elif response == '2':
            return 'date'
        elif response == '3':
            return 'type'
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")


def find_folder_by_name_and_parent(user_id, folder_name, parent_folder_id=None):
    """
    폴더 검색용 API가 있다고 가정하고,
    없으면 None을 리턴.
    """
    # 예시: GET /folder/find?name=folder_name&parentId=parent_folder_id
    # 실제 엔드포인트는 상황에 따라 달라질 수 있음
    print("user_id: {}, folder_name: {}, parent_folder_id: {}".format(user_id, folder_name, parent_folder_id))
    url = f"{BASE_URL}/folder/find"
    folder_request_dto = {
        "name": folder_name,
        "parentFolderId": parent_folder_id,
        "userId": user_id
    }
    response = requests.post(url, json=folder_request_dto)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
    if response.status_code == 200:
        data = response.json()
        print("data : {}", format(data))
        # 만약 data["found"]가 True면 폴더 있음
        if data["found"] == True:
            return data["folderId"] # 폴더가 있으면 그 id 반환
        return None
    else:
        # 검색 실패 시
        return None

def create_folder(user_id, folder_name, parent_folder_id=None):
    """
    새 폴더를 DB에 추가 후 folderId 반환.
    /folder/add 엔드포인트 사용.
    """
    folder_request = {
        "userId": user_id,
        "name": folder_name,
        "parentFolderId": parent_folder_id
    }
    print("folder_request: {}", format(folder_request))
    response = requests.post(f"{BASE_URL}/folder/add", json=folder_request)
    if response.status_code == 200:
        new_folder = response.json()
        return new_folder["id"]
    else:
        raise Exception(f"Folder creation failed. Status={response.status_code}, body={response.text}")

def get_or_create_folder(user_id, folder_name, parent_folder_id=None):
    """
    1) folderName+parentFolderId로 폴더 검색
    2) 없다면 새로 생성
    3) 최종 folderId 반환
    """
    existing_id = find_folder_by_name_and_parent(user_id ,folder_name, parent_folder_id)
    print("existing_id: {}".format(existing_id))
    if existing_id:
        return existing_id
    else:
        return create_folder(user_id, folder_name, parent_folder_id)


def ensure_folder_hierarchy(user_id, full_path):
    """
    full_path를 '/'로 분리해, 상위 폴더부터 차례로 DB에 존재하는지 확인,
    없으면 생성. 최종적으로 leaf 폴더 id를 반환.
    """
    # 1) 슬래시 기준으로 분리 (Windows면 '\\'이 필요할 수 있으므로 os.sep을 써도 좋음)
    parts = [p for p in full_path.split(os.sep) if p]  # 빈 문자열 제거
    current_parent_id = None
    for folder_name in parts:
        print("folder_name: " + folder_name)
        folder_id = get_or_create_folder(user_id, folder_name, current_parent_id)
        current_parent_id = folder_id

    return current_parent_id


def send_operations_to_spring(operations, user_id):
    """
    operations 리스트를 순회하며,
    Spring Boot의 /folder/find|add, /file/upload 엔드포인트로 POST 요청.
    이미 존재하는 폴더라면 새로 생성하지 않음.
    """

    for op in operations:
        source_path = op['source']
        destination_path = op['destination']
        print("path" + source_path +" , " +destination_path)
        # 1) 폴더 경로 추출
        dir_path = os.path.dirname(destination_path)  # 예) /organized_folder/2023/January
        print("dir_path" + dir_path)
        # 2) DB 폴더 생성/검색 -> folderId 획득
        folder_id = ensure_folder_hierarchy(user_id, dir_path)
        print("folder_id: {}", format(folder_id))
        # 3) 파일명, 확장자, 사이즈 추출
        file_name = os.path.basename(destination_path)
        _, ext = os.path.splitext(file_name)
        file_size = os.path.getsize(source_path)

        # 4) FileRequestDTO 준비
        file_request_dto = {
            "userId": user_id,
            "folderId": folder_id,
            "name": file_name,
            "filePath": destination_path,
            "fileType": ext.lower().replace('.', ''),
            "size": file_size
        }
        print(file_request_dto)
        # 5) /file/upload 엔드포인트에 POST
        try:
            response = requests.post(f"{BASE_URL}/file/upload", json=file_request_dto)
            if response.status_code == 200:
                print(f"[SUCCESS] {file_name} 업로드 완료. folderId={folder_id}")
            else:
                print(f"[ERROR] {file_name} 업로드 실패. Status: {response.status_code}, Body: {response.text}")
        except Exception as e:
            print(f"[EXCEPTION] {file_name} 업로드 중 예외 발생: {e}")

def main():
    ensure_nltk_data()

    dry_run = True

    print("-" * 50)
    print("NOTE: Silent mode logs all outputs to a text file instead of displaying them.")
    silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
    if silent_mode:
        log_file = 'operation_log.txt'
    else:
        log_file = None

    while True:
        if not silent_mode:
            print("-" * 50)

        input_path = input("Enter the path of the directory you want to organize: ").strip()
        while not os.path.exists(input_path):
            message = f"Input path {input_path} does not exist. Please enter a valid path."
            if silent_mode and log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
            input_path = input("Enter the path of the directory you want to organize: ").strip()

        message = f"Input path recognized: {input_path}"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)

        output_path = input("Enter the path to store organized files/folders (press Enter to use 'organized_folder'): ").strip()
        if not output_path:
            output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

        message = f"Output path set to: {output_path}"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)

        start_time = time.time()
        file_paths = collect_file_paths(input_path)
        end_time = time.time()

        message = f"Time taken to load file paths: {end_time - start_time:.2f} seconds"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)
            print("Directory tree before organizing:")
            display_directory_tree(input_path)
            print("*" * 50)

        while True:
            mode = get_mode_selection()

            if mode == 'content':
                # Separate images and text
                image_files, text_files = separate_files_by_type(file_paths)

                # Prepare text input for GPT
                text_tuples = []
                for fp in text_files:
                    text_content = read_file_data(fp)
                    if text_content is None:
                        msg = f"Unsupported or unreadable text file: {fp}"
                        if silent_mode and log_file:
                            with open(log_file, 'a') as f:
                                f.write(msg + '\n')
                        else:
                            print(msg)
                        continue
                    text_tuples.append((fp, text_content))

                # (2) Process images with GPT-based approach (requires image_url_map)
                data_images = process_image_files(
                    image_files,
                    silent=silent_mode,
                    log_file=log_file
                )

                # (3) Process text files with GPT
                data_texts = process_text_files(
                    text_tuples,
                    silent=silent_mode,
                    log_file=log_file
                )

                renamed_files = set()
                processed_files = set()
                all_data = data_images + data_texts

                operations = compute_operations(
                    all_data,
                    output_path,
                    renamed_files,
                    processed_files
                )

            elif mode == 'date':
                operations = process_files_by_date(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )
                print(operations)
            elif mode == 'type':
                operations = process_files_by_type(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )
                print(operations)
            else:
                print("Invalid mode selected.")
                return

            print("-" * 50)
            message = "Proposed directory structure:"
            if silent_mode and log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
                print(os.path.abspath(output_path))

                def simulate_tree(ops, base):
                    t = {}
                    for op in ops:
                        rel_path = os.path.relpath(op['destination'], base)
                        parts = rel_path.split(os.sep)
                        node = t
                        for p in parts:
                            if p not in node:
                                node[p] = {}
                            node = node[p]
                    return t

                def print_tree(tree, prefix=''):
                    keys = list(tree.keys())
                    for i, k in enumerate(keys):
                        symbol = '└── ' if i == len(keys) - 1 else '├── '
                        print(prefix + symbol + k)
                        if tree[k]:
                            extension = '    ' if i == len(keys) - 1 else '│   '
                            print_tree(tree[k], prefix + extension)

                simulated = simulate_tree(operations, output_path)
                print_tree(simulated)
                print("-" * 50)

            proceed = get_yes_no("Would you like to proceed with these changes? (yes/no): ")
            if proceed:
                os.makedirs(output_path, exist_ok=True)
                msg = "Performing file operations..."
                if silent_mode and log_file:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                else:
                    print(msg)

                execute_operations(
                    operations,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )
                user_id = 1 # 예시 user_id는 spring에서 받아오기
                send_operations_to_spring(operations, user_id)

                msg = "The files have been organized successfully."
                if silent_mode and log_file:
                    with open(log_file, 'a') as f:
                        f.write("-" * 50 + '\n' + msg + '\n' + "-" * 50 + '\n')
                else:
                    print("-" * 50)
                    print(msg)
                    print("-" * 50)
                break
            else:
                another_sort = get_yes_no("Would you like to choose another sorting method? (yes/no): ")
                if another_sort:
                    continue
                else:
                    print("Operation canceled by the user.")
                    break

        another_directory = get_yes_no("Would you like to organize another directory? (yes/no): ")
        if not another_directory:
            break



if __name__ == '__main__':
    main()
