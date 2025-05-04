import os
import time
import nltk
import requests
import tempfile

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_file_data, separate_files_by_type_from_metadata
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

def get_folder_data(folder_id, user_id):
    url = f"{BASE_URL}/folder/hierarchy/{folder_id}/{user_id}"
    r = requests.get(url)
    r.raise_for_status()
    print("r.json(): ", format(r.json()))
    return r.json()

def mark_folder_deleted(user_id, folder_id):
    url = f"{BASE_URL}/folder/delete"  # 서버와 협의된 엔드포인트
    payload = {
        "userId": user_id,
        "folderId": folder_id,
        "isDeleted": True
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"[SUCCESS] Folder {folder_id} marked as deleted in DB.")
        else:
            print(f"[ERROR] Could not mark folder {folder_id} as deleted. Status={response.status_code}, Body={response.text}")
    except Exception as e:
        print(f"[EXCEPTION] While marking folder {folder_id} as deleted: {e}")

def extract_files_from_tree(folder_tree):
    """
    폴더 트리 JSON(예: FolderDTO 구조)에서 모든 파일의 정보를 재귀적으로 추출한다.
    각 파일 정보는 {"file_path": ..., "fileType": ..., "name": ...} 구조로 구성.
    """
    files = []
    if "files" in folder_tree:
        for f in folder_tree["files"]:
            s3_url = f.get("fileUrl")
            local_path = download_file_from_url(s3_url) if s3_url else None
            files.append({
                "fileId": f.get("id"),
                "file_path": local_path,
                "fileType": f.get("fileType", "").lower(),
                "name": f.get("name"),
                "createdAt": f.get("createdAt"),
                "size": f.get("size"),
                "fileUrl": s3_url
            })
    if "subFolders" in folder_tree:
        for sub in folder_tree["subFolders"]:
            files.extend(extract_files_from_tree(sub))
    return files

def do_auto_classification(folder_tree, destination_folder_id, mode="type", output_path="/organized"):
    """
    folder_tree: Spring Boot에서 전달받은 폴더 트리 JSON (FolderDTO 구조)
    mode: "content", "date", "type" 중 하나
         - content: 파일 내용 기반 (GPT로 이미지와 텍스트 각각 처리)
         - date: 파일 수정일 등 날짜 기준 분류
         - type: 파일 확장자 등 유형 기준 분류
    output_path: 새로 정리할 기준 루트 경로

    반환: OrganizedResultDTO 형태의 dict
             { "folderId": ..., "summary": ..., "operations": [...] }
    """
    # 1) folder_tree로부터 모든 파일 정보 추출 (재귀적으로)
    print("folder_tree: {}".format(folder_tree))
    file_list = extract_files_from_tree(folder_tree)
    print("file_list: {}".format(file_list))
    operations = []
    # 각 모드에 따라 기존 구현 함수 호출
    if mode == "content":
        # content 모드: GPT 기반 이미지/텍스트 분리 처리
        file_paths = [f["file_path"] for f in file_list if f["file_path"]]
        print("file_paths: {}".format(file_paths))
        # 분류: 이미지와 텍스트 파일 분리 (이미 존재하는 separate_files_by_type 사용)
        image_files, text_files = separate_files_by_type_from_metadata(file_list)
        print("image_files: {}".format(image_files))
        print("text_files: {}".format(text_files))
        # 텍스트 파일의 경우 실제 내용 읽기
        text_tuples = []
        for fp in text_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                    print("success Reading: {}".format(fp))
            except Exception as e:
                print(f"[ERROR] Failed to read text from {fp}: {e}")
                text_content = None
            if text_content:
                text_tuples.append((fp, text_content))

        data_images = process_image_files(image_files, file_list=file_list, silent=True, log_file=None)
        data_texts = process_text_files(text_tuples, file_list=file_list, silent=True, log_file=None)
        all_data = data_images + data_texts
        renamed_files = set()
        processed_files = set()
        operations = compute_operations(all_data, output_path, renamed_files, processed_files)
        for f in file_list:
            local_path = f.get("file_path")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"[CLEANUP] Deleted temp file: {local_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {local_path}: {e}")
    elif mode == "date":
        # date 모드: 날짜 기준 분류
        #file_paths = [f["file_path"] for f in file_list if f["file_path"]]
        operations = process_files_by_date(file_list, output_path, dry_run=False, silent=True, log_file=None)

    elif mode == "type":
        # type 모드: 확장자, 파일 유형 기준 분류
        # file_paths = [f["file_path"] for f in file_list if f["file_path"]]
        operations = process_files_by_type(file_list, output_path, dry_run=False, silent=True, log_file=None)
    else:
        raise ValueError("Invalid mode. Choose 'content', 'date', or 'type'.")

    overall_summary = f"Auto-classification ({mode} mode) completed for folder '{folder_tree.get('name')}'. {len(operations)} files processed."

    result_dict = {
        "folderId": folder_tree.get("id"),
        "summary": overall_summary,
        "destinationFolderId": destination_folder_id,
        "operations": operations
    }
    print("result: {}", format(result_dict))
    print("operation: {} ", format(operations))
    return result_dict

def download_file_from_url(url):
    """
    S3 URL로부터 파일을 다운로드하고, 임시 로컬 경로를 반환한다.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    else:
        print(f"[ERROR] Failed to download file: {url}")
        return None

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
    print("NOTE: Silent mode logs outputs to a text file instead of displaying them.")
    silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
    log_file = 'operation_log.txt' if silent_mode else None

    # 여기서 추가: UI 방식으로 폴더 선택 시, 폴더 경로 대신 Spring DB의 folderId로부터
    # 폴더 트리 정보를 가져와 Python에서 자동분류를 시작하는 옵션을 둘 수 있음.
    use_db_folder = get_yes_no("Do you want to organize based on a DB folder? (yes/no): ")
    if use_db_folder:
        folder_id = int(input("Enter the DB folder ID to organize: "))
        folder_tree = get_folder_data(folder_id)

        # do_auto_classification(folder_tree)는 위에서 구현한 함수로, 폴더 트리를 기반으로 파일 이동 등 작업을 결정
        result_dict = do_auto_classification(folder_tree) #, mode, output_path)
        print("Auto-classification Result:")
        print(result_dict)
        # 최종 결과를 Spring에 전송
        spring_response = requests.post(f"{BASE_URL}/organize/result", json=result_dict)
        if spring_response.status_code == 200:
            try:
                spring_resp = spring_response.json()
            except Exception as e:
                spring_resp = spring_response.text
            print("Final result sent to Spring:", spring_resp)
        else:
            print("Error sending final result to Spring:", spring_response.status_code, spring_response.text)
        return  # DB 기반 분류 완료 후 종료

    # 기존 로컬 경로 처리
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

        # 모드 선택: Content, Date, Type
        mode = get_mode_selection()
        if mode == 'content':
            image_files, text_files = separate_files_by_type(file_paths)
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
            data_images = process_image_files(image_files, silent=silent_mode, log_file=log_file)
            data_texts = process_text_files(text_tuples, silent=silent_mode, log_file=log_file)
            renamed_files = set()
            processed_files = set()
            all_data = data_images + data_texts
            operations = compute_operations(all_data, output_path, renamed_files, processed_files)
        elif mode == 'date':
            operations = process_files_by_date(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
            print(operations)
        elif mode == 'type':
            operations = process_files_by_type(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
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
            execute_operations(operations, dry_run=False, silent=silent_mode, log_file=log_file)
            user_id = 1  # 예시 user_id, 실제는 Spring에서 받아옴
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
