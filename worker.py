import os
import json
import time
import logging
import re
import boto3
import requests
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
import tempfile
import subprocess
import hashlib

logging.basicConfig(level=logging.INFO)

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
QUEUE_URL = os.getenv("THUMB_SQS_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/"

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://localhost:8080")
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "12345678")

LIBREOFFICE_PATH = os.getenv("LIBREOFFICE_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")

sqs = boto3.client("sqs", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)

def create_text_thumbnail(text, width=240, height=240):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 10
    for line in text.splitlines()[:12]:
        draw.text((10, y), line[:40], fill="black", font=font)
        y += 16
    return img

def download_file(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def generate_thumbnail_bytes(file_bytes: bytes, file_name: str) -> bytes:
    ext = file_name.split(".")[-1].lower()
    base_name = file_name.rsplit(".", 1)[0]
    hash_part = hashlib.md5(file_name.encode()).hexdigest()[:8]
    thumb_name = f"thumb_{base_name}_{hash_part}.jpg"

    img = None

    if ext in ["png", "jpg", "jpeg"]:
        img = Image.open(BytesIO(file_bytes))

    elif ext == "pdf":
        img = convert_from_bytes(file_bytes)[0]

    elif ext in ["txt", "md", "csv"]:
        text = file_bytes.decode("utf-8", errors="ignore")[:1000]
        img = create_text_thumbnail(text)

    elif ext in ["doc", "docx", "xls", "xlsx", "ppt", "pptx"]:
        if not LIBREOFFICE_PATH or not POPPLER_PATH:
            raise RuntimeError("LIBREOFFICE_PATH or POPPLER_PATH missing")

        safe_name = safe_filename(file_name)
        safe_base = safe_filename(base_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, safe_name)
            pdf_path = os.path.join(tmpdir, safe_base + ".pdf")

            with open(input_path, "wb") as f:
                f.write(file_bytes)

            subprocess.run(
                [
                    LIBREOFFICE_PATH,
                    "--invisible",
                    "--convert-to", "pdf",
                    "--outdir", tmpdir,
                    input_path
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if not os.path.exists(pdf_path):
                raise RuntimeError("PDF convert failed")

            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=1,
                size=(240, 240),
                poppler_path=POPPLER_PATH
            )
            img = images[0]

    else:
        raise RuntimeError("unsupported file type")

    img.thumbnail((240, 240))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    return thumb_name, buf.getvalue()

def upload_thumbnail_to_s3(thumb_name: str, thumb_bytes: bytes) -> str:
    s3.upload_fileobj(
        BytesIO(thumb_bytes),
        S3_BUCKET,
        f"thumbnails/{thumb_name}",
        ExtraArgs={"ContentType": "image/jpeg"}
    )
    return f"{S3_BASE_URL}thumbnails/{thumb_name}"

def update_spring(file_id: int, thumbnail_url: str):
    url = f"{SPRING_BASE_URL}/internal/files/{file_id}/thumbnail"
    headers = {"X-INTERNAL-TOKEN": INTERNAL_TOKEN}
    payload = {"thumbnailUrl": thumbnail_url}
    r = requests.patch(url, json=payload, headers=headers, timeout=10)
    logging.info("update_spring status=%s body=%s", r.status_code, r.text[:200])
    r.raise_for_status()

def process_message(msg):
    body = json.loads(msg["Body"])
    file_id = body["fileId"]
    file_url = body["fileUrl"]
    file_name = body["fileName"]

    logging.info("start thumbnail fileId=%s fileName=%s", file_id, file_name)

    file_bytes = download_file(file_url)
    thumb_name, thumb_bytes = generate_thumbnail_bytes(file_bytes, file_name)
    thumb_url = upload_thumbnail_to_s3(thumb_name, thumb_bytes)

    update_spring(file_id, thumb_url)

    logging.info("done thumbnail fileId=%s", file_id)

def run():
    if not QUEUE_URL:
        raise RuntimeError("THUMB_SQS_URL missing")
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET missing")

    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )

        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        msg = msgs[0]
        receipt = msg["ReceiptHandle"]

        try:
            process_message(msg)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
        except Exception as e:
            logging.exception("failed: %s", e)
            # 삭제 안 하면 Visibility Timeout 이후 재시도됨
            # DLQ 설정했으면 실패 누적 시 DLQ로 이동

if __name__ == "__main__":
    run()
