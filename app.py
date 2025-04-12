# app.py
import os
import requests
import openai
from flask import Flask, request, jsonify
from text_data_processing import process_single_text_file
from image_data_processing import process_single_image
from data_processing_common import sanitize_filename
from main import get_folder_data, do_auto_classification  # 예: main.py에 구현

app = Flask(__name__)

BASE_URL = "http://localhost:8080"
@app.route('/organize_folder', methods=['POST'])
def organize_folder():
    """
    1) Spring으로부터 { "folderId": 2, "mode": "content" } 등의 데이터를 받음
    2) get_folder_data(folderId)로 폴더 트리 JSON을 조회
    3) do_auto_classification(folder_tree, mode) 호출
    4) 결과 OrganizeResultDTO를 Spring의 /organize/result로 전송
    """
    data = request.json
    folder_id = data['folderId']
    mode = data.get("mode", "content")  # mode 기본값은 "content"

    # A) 폴더 트리 조회 (Spring API 호출)
    folder_tree = get_folder_data(folder_id)

    # B) 자동 분류 실행 (mode에 따라)
    result_dict = do_auto_classification(folder_tree, mode=mode, output_path="/organized")
    print("Auto-classification result:")
    print(result_dict)

    # C) 결과 Spring으로 전송
    response = requests.post(f"{BASE_URL}/organize/result", json=result_dict)
    if response.status_code == 200:
        return jsonify({"message": "Organize done", "springResponse": response.json()})
    else:
        return jsonify({"message": "Organize done, but error sending to spring",
                        "status": response.status_code}), 500


@app.route('/classify', methods=['POST'])
def classify_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir = './uploads'
    os.makedirs(temp_dir, exist_ok=True)
    saved_path = os.path.join(temp_dir, file.filename)
    file.save(saved_path)

    ext = os.path.splitext(file.filename.lower())[1]
    classification_result = {}

    # 이미지
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
        result = process_single_image(
            image_path=saved_path,
            api_key=os.getenv("OPENAI_API_KEY"),
            silent=True
        )
        classification_result = {
            "filePath": result["file_path"],
            "recommendedFolder": result["foldername"],
            "recommendedFilename": result["filename"],
            "summary": result["description"]
        }

    # 텍스트/문서
    elif ext in ['.txt', '.doc', '.docx', '.pdf', '.md', '.xls', '.xlsx', '.ppt', '.pptx', '.csv']:
        text_content = file.read().decode('utf-8', errors='ignore')
        processed = process_single_text_file(
            (saved_path, text_content),
            silent=True
        )
        classification_result = {
            "filePath": processed["file_path"],
            "recommendedFolder": processed["foldername"],
            "recommendedFilename": processed["filename"],
            "summary": processed["description"]
        }
    else:
        classification_result = {
            "filePath": saved_path,
            "recommendedFolder": "others",
            "recommendedFilename": sanitize_filename(file.filename),
            "summary": "Unknown file type"
        }

    return jsonify(classification_result), 200

if __name__ == '__main__':
    # 실제 운영 시 gunicorn 등으로 배포
    app.run(host='0.0.0.0', port=5000, debug=True)
