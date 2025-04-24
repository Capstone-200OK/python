# app.py
import os
import requests
import openai
import boto3
from io import BytesIO
from flask import Flask, request, jsonify
from text_data_processing import process_single_text_file
from image_data_processing import process_single_image
from data_processing_common import sanitize_filename
from main import get_folder_data, do_auto_classification  # 예: main.py에 구현
from werkzeug.utils import secure_filename
from PIL import Image
from pdf2image import convert_from_bytes
from pdf2image import convert_from_path
import tempfile
import subprocess

app = Flask(__name__)

libreoffice_path = os.getenv("LIBREOFFICE_PATH")
poppler_path=os.getenv("POPPLER_PATH")
BASE_URL = "http://localhost:8080"
S3_BUCKET = os.getenv("S3_BUCKET")  # 수정
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BASE_URL = f"https://{S3_BUCKET}.s3.ap-northeast-2.amazonaws.com/"
s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name='ap-northeast-2'
)

@app.route('/organize_folder', methods=['POST'])
def organize_folder():
    """
    1) Spring으로부터 { "folderId": 2, "mode": "content" } 등의 데이터를 받음
    2) get_folder_data(folderId)로 폴더 트리 JSON을 조회
    3) do_auto_classification(folder_tree, mode) 호출
    4) 결과 OrganizeResultDTO를 Spring의 /organize/result로 전송
    """
    data = request.json
    folder_ids = data['folderIds']
    mode = data['mode']  # mode 기본값은 "content"
    output_path = data['output_path']
    destination_folder_id = data['destinationFolderId']
    # A) 폴더 트리 조회 (Spring API 호출)
    if len(folder_ids) == 1:
        folder_tree = get_folder_data(folder_ids[0])
    else :
        folder_tree = {
            "id": None,
            "name": None,
            "files": [],
            "subFolders": [get_folder_data(fid) for fid in folder_ids],
            "isDeleted": False
        }
    # B) 자동 분류 실행 (mode에 따라)
    result_dict = do_auto_classification(folder_tree,destination_folder_id,mode=mode, output_path=output_path)
    result_dict["sourceFolderIds"] = folder_ids
    print("Auto-classification result:")
    print(result_dict)

    # C) 결과 Spring으로 전송
    response = requests.post(f"{BASE_URL}/organize/result", json=result_dict)
    if response.status_code == 200:
        return jsonify({"message": "Organize done", "springResponse": response.json()})
    else:
        return jsonify({"message": "Organize done, but error sending to spring",
                        "status": response.status_code}), 500




@app.route('/api/thumbnail', methods=['POST'])
def generate_thumbnail():
    print("[DEBUG] LibreOffice 경로:", libreoffice_path)
    data = request.json
    file_url = data['fileUrl']
    file_name = secure_filename(data['fileName'])

    ext = file_name.split('.')[-1].lower()
    thumb_name = f"thumb_{file_name.rsplit('.', 1)[0]}.jpg"

    # 이미지 썸네일 처리 (doc/pdf 처리도 가능)
    response = requests.get(file_url)
    img = None
    if ext in ['png', 'jpg', 'jpeg']:
        img = Image.open(BytesIO(response.content))
    elif ext == 'pdf':
        img = convert_from_bytes(response.content)[0]
    elif ext in ['docx', 'xlsx', 'pptx']:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, file_name)
            pdf_path = os.path.join(tmpdir, file_name.rsplit('.', 1)[0] + ".pdf")

            with open(input_path, 'wb') as f:
                f.write(response.content)

            try:
                result = subprocess.run([
                    libreoffice_path,
                    "--invisible",
                    "--convert-to", "pdf",
                    "--outdir", tmpdir,
                    input_path
                ], check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("[LibreOffice stdout]", result.stdout)
                print("[LibreOffice stderr]", result.stderr)
                print("[DEBUG] Expecting PDF at:", pdf_path)
            except subprocess.CalledProcessError as e:
                print("[LibreOffice 오류]", e.stderr)
                return jsonify({"error": f"LibreOffice 변환 실패: {e.stderr}"}), 500

            if not os.path.exists(pdf_path):
                return jsonify({"error": "PDF 생성 실패"}), 500

            try:
                images = convert_from_path(pdf_path, first_page=1, last_page=1, size=(240, 240),poppler_path=poppler_path)
                img = images[0]
            except Exception as e:
                return jsonify({"error": f"이미지 변환 실패: {str(e)}"}), 500


    img.thumbnail((240, 240))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    s3.upload_fileobj(buf, S3_BUCKET, f"thumbnails/{thumb_name}", ExtraArgs={'ContentType': 'image/jpeg'})

    return jsonify({"thumbnailUrl": f"{S3_BASE_URL}thumbnails/{thumb_name}"})

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
