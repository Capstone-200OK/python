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
import hashlib
from PIL import Image, ImageDraw, ImageFont

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
font_path = os.path.join("fonts", "NotoSansKR-Regular.ttf")
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
    user_id = data['userId']
    # A) 폴더 트리 조회 (Spring API 호출)
    print("folder_ids: ", folder_ids)
    if len(folder_ids) == 1:
        print("folderLen: ", format(len(folder_ids)))
        folder_tree = get_folder_data(folder_ids[0], user_id)
    else :
        print("folderLen: ", format(len(folder_ids)))
        folder_tree = {
            "id": None,
            "name": None,
            "files": [],
            "subFolders": [get_folder_data(fid, user_id) for fid in folder_ids],
            "isDeleted": False
        }
    # B) 자동 분류 실행 (mode에 따라)
    result_dict = do_auto_classification(folder_tree,destination_folder_id,mode=mode, output_path=output_path)
    result_dict["sourceFolderIds"] = folder_ids
    result_dict["userId"] = user_id
    print("✅ isScheduled 전달됨:", data.get("isScheduled"))
    result_dict["isScheduled"] = data.get("isScheduled", False)
    result_dict["originalStartFolderIds"] = folder_ids
    result_dict["isMaintain"] = data.get("isMaintain", False)
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

    base_name = file_name.rsplit('.', 1)[0]
    ext = file_name.split('.')[-1].lower()
    hash_part = hashlib.md5(file_name.encode()).hexdigest()[:8]
    thumb_name = f"thumb_{base_name}_{hash_part}.jpg"

    response = requests.get(file_url)
    img = None

    if ext in ['png', 'jpg', 'jpeg']:
        img = Image.open(BytesIO(response.content))

    elif ext == 'pdf':
        img = convert_from_bytes(response.content)[0]

    elif ext in ['txt', 'md', 'csv']:
        response.encoding = 'utf-8'
        text = response.text[:1000]
        img = create_text_thumbnail(text)

    elif ext in ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx']:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, file_name)
            pdf_path = os.path.join(tmpdir, base_name + ".pdf")

            with open(input_path, 'wb') as f:
                f.write(response.content)

            try:
                result = subprocess.run([
                    libreoffice_path,
                    "--invisible",
                    "--convert-to", "pdf",
                    "--outdir", tmpdir,
                    input_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("[LibreOffice stdout]", result.stdout)
                print("[LibreOffice stderr]", result.stderr)
            except subprocess.CalledProcessError as e:
                print("[LibreOffice 오류]", e.stderr)
                return jsonify({"error": f"LibreOffice 변환 실패: {e.stderr}"}), 500

            if not os.path.exists(pdf_path):
                return jsonify({"error": "PDF 생성 실패"}), 500

            try:
                images = convert_from_path(pdf_path, first_page=1, last_page=1, size=(240, 240), poppler_path=poppler_path)
                img = images[0]
            except Exception as e:
                return jsonify({"error": f"이미지 변환 실패: {str(e)}"}), 500

    if img is None:
        return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400

    # 썸네일 크기 적용 및 모드 변환
    img.thumbnail((240, 240))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    s3.upload_fileobj(buf, S3_BUCKET, f"thumbnails/{thumb_name}", ExtraArgs={'ContentType': 'image/jpeg'})

    return jsonify({"thumbnailUrl": f"{S3_BASE_URL}thumbnails/{thumb_name}"})

def create_text_thumbnail(text, width=240, height=240, font_size=14):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[WARN] 사용자 폰트 로딩 실패, 기본 폰트 사용: {e}")
        font = ImageFont.load_default()

    lines = []
    words = text.split()
    line = ''
    for word in words:
        # line이 비어있을 경우를 고려한 테스트 문자열 생성
        test_line = (line + ' ' + word).strip() if line else word
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
        except Exception as e:
            print(f"[ERROR] 텍스트 bbox 계산 실패: {e}")
            text_width = width

        if text_width < width - 20:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = 10
    for line in lines[:10]:
        draw.text((10, y), line, fill='black', font=font)
        y += font_size + 4

    return img

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
    app.run(host='0.0.0.0', port=5050, debug=True)
