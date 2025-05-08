import os
import re
import datetime  # Import datetime for date operations
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from dateutil import parser

def strip_category_prefix(category_line: str) -> str:
    if category_line.lower().startswith("category:"):
        return category_line.split(":", 1)[1].strip()
    return category_line.strip()

def sanitize_filename(name, max_length=50, max_words=5):
    """Sanitize the filename by removing unwanted words and characters."""
    # Remove file extension if present
    name = os.path.splitext(name)[0]

    name = re.sub(r'^(filename_|file_|document_|summary_|output_|result_)', '', name, flags=re.IGNORECASE)
    # Remove unwanted words and data type words
    name = re.sub(
        r'\b(jpg|jpeg|png|gif|bmp|txt|md|pdf|docx|xls|xlsx|csv|ppt|pptx|image|picture|photo|this|that|these|those|here|there|'
        r'please|note|additional|notes|folder|name|sure|heres|a|an|the|and|of|in|'
        r'to|for|on|with|your|answer|should|be|only|summary|summarize|text|category)\b',
        '',
        name,
        flags=re.IGNORECASE
    )
    # Remove non-word characters except underscores
    sanitized = re.sub(r'[^\w\s]', '', name).strip()
    # Replace multiple underscores or spaces with a single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Convert to lowercase
    sanitized = sanitized.lower()
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Split into words and limit the number of words
    words = sanitized.split('_')
    limited_words = [word for word in words if word]  # Remove empty strings
    limited_words = limited_words[:max_words]
    limited_name = '_'.join(limited_words)
    # Limit length
    return limited_name[:max_length] if limited_name else 'untitled'

def process_files_by_date(file_infos, output_path, dry_run=False, silent=False, log_file=None):
    """Process files to organize them by date."""
    operations = []
    for file_info in file_infos:
        file_path = file_info.get('file_path')
        # Get the modification time
        mod_time = file_info.get('createdAt')
        # Convert to datetime
        mod_datetime = parser.parse(mod_time)
        year = mod_datetime.strftime('%Y') # e.g. ,'2025'
        month = mod_datetime.strftime('%m')  # e.g., %B -> 'January','%m' -> 01
        # Create directory path
        dir_path = "/".join([output_path, year, month]) # "/"로 구분하여 join
        # Prepare new file path
        new_file_name = os.path.basename(file_path) # 파일명 반환
        new_file_path = "/".join([dir_path, new_file_name])
        # Decide whether to use hardlink or symlink
        link_type = 'hardlink'  # Assume hardlink for now
        # Record the operation
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
            'fileId': file_info.get("fileId"),
            'name': file_info.get("name"),
            'fileType': file_info.get("fileType"),
            'size': file_info.get("size"),
            "createdAt": mod_time,
        }
        operations.append(operation)
    return operations

def process_files_by_type(file_infos, output_path, dry_run=False, silent=False, log_file=None):
    """Process files to organize them by type, first separating into text-based and image-based files."""
    operations = []

    # Define extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    text_extensions = ('.txt', '.md', '.docx', '.doc', '.pdf', '.xls', '.xlsx', '.epub', '.mobi', '.azw', '.azw3')

    for file_info in file_infos:
        file_path = file_info.get("file_path")
        # Exclude hidden files (additional safety)
        if os.path.basename(file_path).startswith('.'):
            continue

        # Get the file extension
        ext = os.path.splitext(file_path)[1].lower()

        # Check if it's an image file
        if ext in image_extensions:
            # Image-based files
            top_folder = 'image_files'
            # You can add subcategories here if needed
            folder_name = top_folder

        elif ext in text_extensions:
            # Text-based files
            top_folder = 'text_files'
            # Map extensions to subfolders
            if ext in ('.txt', '.md'):
                sub_folder = 'plain_text_files'
            elif ext in ('.doc', '.docx'):
                sub_folder = 'doc_files'
            elif ext == '.pdf':
                sub_folder = 'pdf_files'
            elif ext in ('.xls', '.xlsx'):
                sub_folder = 'xls_files'
            elif ext in ('.epub', '.mobi', '.azw', '.azw3'):
                sub_folder = 'ebooks'
            else:
                sub_folder = 'others'
            folder_name = "/".join([top_folder, sub_folder])

        else:
            # Other types
            folder_name = 'others'

        # Create directory path
        dir_path = "/".join([output_path.strip("/"), folder_name.strip("/")])
        # Prepare new file path
        new_file_name = os.path.basename(file_path)
        new_file_path = "/".join([dir_path, new_file_name])
        # Decide whether to use hardlink or symlink
        link_type = 'hardlink'  # Assume hardlink for now
        # Record the operation
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
            'fileId': file_info.get("fileId"),
            'name': file_info.get("name"),
            'fileType': file_info.get("fileType"),
            'size': file_info.get("size")
        }
        operations.append(operation)

    return operations

def compute_operations(file_name_change, data_list, new_path, renamed_files, processed_files):
    """Compute the file operations based on generated metadata."""
    operations = []
    for file_info in data_list:
        file_path = file_info['file_path']
        if file_path in processed_files:
            continue
        processed_files.add(file_path)

        # 폴더 이름과 새로운 파일 이름 구성
        folder_name = file_info['category']
        extension = os.path.splitext(file_info.get("name"))[1]  # 진짜 원래 이름 기준
        if not file_name_change:
            base_filename = file_info.get("filename") or os.path.splitext(file_info.get("name"))[0]
        else :
            base_filename = os.path.splitext(file_info.get("name"))[0]
        new_file_name = base_filename + extension

        # 디렉토리 및 새 경로 구성
        dir_path = "/".join([new_path, folder_name])
        new_file_path = "/".join([dir_path, new_file_name])

        # 중복 방지
        counter = 1
        while new_file_path in renamed_files:
            new_file_name = f"{base_filename}_{counter}{extension}"
            new_file_path = "/".join([dir_path, new_file_name])
            counter += 1

        renamed_files.add(new_file_path)

        # 링크 방식 결정 (필요시 symlink 지원 가능)
        link_type = 'hardlink'

        # operation에 메타데이터 추가
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
            'fileId': file_info.get("fileId"),
            'name': new_file_name,
            'fileType': file_info.get("fileType"),
            'size': file_info.get("size"),
            'createdAt': file_info.get("createdAt"),
        }
        operations.append(operation)

    return operations

def execute_operations(operations, dry_run=False, silent=False, log_file=None):
    """Execute the file operations."""
    total_operations = len(operations)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Organizing Files...", total=total_operations)
        for operation in operations:
            source = operation['source']
            destination = operation['destination']
            link_type = operation['link_type']
            dir_path = os.path.dirname(destination)

            if dry_run:
                message = f"Dry run: would create {link_type} from '{source}' to '{destination}'"
            else:
                # Ensure the directory exists before performing the operation
                os.makedirs(dir_path, exist_ok=True)

                try:
                    if link_type == 'hardlink':
                        os.link(source, destination)
                    else:
                        os.symlink(source, destination)
                    message = f"Created {link_type} from '{source}' to '{destination}'"
                except Exception as e:
                    message = f"Error creating {link_type} from '{source}' to '{destination}': {e}"

            progress.advance(task)

            # Silent mode handling
            if silent:
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(message + '\n')
            else:
                print(message)