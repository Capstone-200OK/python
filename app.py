# app.py
import os
from flask import Flask, request, jsonify
from text_data_processing import process_single_text_file
from image_data_processing import process_single_image
from data_processing_common import sanitize_filename

app = Flask(__name__)

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
