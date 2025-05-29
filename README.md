# Local File Organizer: AI File Management Run Entirely on Your Device, Privacy Assured

Tired of digital clutter? Overwhelmed by disorganized files scattered across your computer? Let AI do the heavy lifting! The Local File Organizer is your personal organizing assistant, using cutting-edge AI to bring order to your file chaos - all while respecting your privacy.

## How It Works 💡

Before:

```
Personal/뻐꾸기
├── 콘텐츠 기획안.pdf
├── 참고 논문 및 기사.pdf
├── 영자 신문 기사 스크랩.docx
├── 시사용어 정리.docx
├── 아이디어 노트.docx
├── 팀플 발표 자료.pptx
├── 문학 작품 분석.docx
├── 할 일 리스트.xlsx
├── 뉴스 트렌드 정리.xlsx
├── 기사 감상문.txt
├── 감상일지.txt
└── 팀플 회의록.txt

0 directories, 12 files
```

After:

```
Personal/보관함
├── Business
│   └── 콘텐츠 기획안.pdf
├── News
│   ├── 영자 신문 기사 스크랩.docx
|   └── 뉴스 트랜드 정리.xlsx
├── Instruction
│   └── 시사용어 정리.docx
├── Creative
│   └── 아이디어 노트.docx
├── Academic
│   ├── 팀플 발표 자료.pptx
│   ├── 문학 작품 분석.docx
│   ├── 할 일 리스트.xlsx
│   ├── 기사 감상문.txt
│   └── 참고 논문 및 기사.pdf
├── General Document
│   └── 팀플 회의록.txt
└── Personal
    └── 감상일지.txt

7 directories, 12 files
```

## Basic enviroment 

- Anaconda prompt
- pycharm

## Supported File Types 📁

- **Images:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- **Text Files:** `.txt`, `.docx`, `.md`
- **Spreadsheets:** `.xlsx`, `.csv`
- **Presentations:** `.ppt`, `.pptx`
- **PDFs:** `.pdf`

### 1. Install Python

Before installing the Local File Organizer, make sure you have Python installed on your system. We recommend using Python 3.12 or later.

You can download Python from [the official website]((https://www.python.org/downloads/)).

Follow the installation instructions for your operating system.

### 2. Clone the Repository

Clone this repository to your local machine using Git:

```zsh
git clone  https://github.com/Capstone-200OK/python.git
```

Or download the repository as a ZIP file and extract it to your desired location.

### 3. Set Up the Python Environment

Create a new Conda environment named `local_file_organizer` with Python 3.12:

```zsh
conda create --name local_file_organizer python=3.12
```

Activate the environment:

```zsh
conda activate local_file_organizer
```

### 4. Install Dependencies 

1. Ensure you are in the project directory:
   ```zsh
   cd path/to/Local-File-Organizer
   ```
   Replace `path/to/Local-File-Organizer` with the actual path where you cloned or extracted the project.

2. Install the required dependencies:
   ```zsh
   pip install -r requirements.txt
   ```

With the environment activated and dependencies installed, run the script using:

### 5. Running the Script🎉
```zsh
python app.py
```

## License

This project is dual-licensed under the MIT License and Apache 2.0 License. You may choose which license you prefer to use for this project.

- See the [MIT License](LICENSE-MIT) for more details.
