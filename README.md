# Nobelize

**AI 기반의 글 평가/수정 서비스, “**Nobelize**”**

- “Nobel(노벨 문학상) + Realize(실현하다)”
- AI와 Docker/Kubernetes 클라우드 시스템을 활용한 글 평가/수정 서비스.

#

### 폴더 구조

```plaintext
Nobelize_cloud/
├── main.py                  # FastAPI 서버 실행 코드
├── utils.py                 # FAISS 및 OpenAI 관련 유틸리티 함수
├── embeddings/              # FAISS 인덱스 저장
│   └── faiss_index3.index
├── data/                    # PDF 파일 저장
│   └── example.pdf          # 테스트용 PDF 파일
├── requirements.txt         # 필요한 패키지 목록
├── .env                     # 환경 변수 파일 (OpenAI API Key)
└── .gitignore               # Git 무시 파일 ( .env 등)
```

### **1. Python 가상환경**

1. 가상환경 생성

   ```bash
   python -m venv venv
   ```

2. 가상환경 활성화

   - Windows (CMD):

     ```bash
     venv\Scripts\activate
     ```

   - Windows (PowerShell):

     ```powershell
     .\venv\Scripts\activate
     ```

   - Linux/MacOS:
     ```bash
     source venv/bin/activate
     ```

3. 가상환경 비활성화
   ```bash
   deactivate
   ```

### **2. `requirements.txt`**

```bash
pip install -r requirements.txt
```

### **3. `실행`**

```bash
.\venv\Scripts\activate
uvicorn main:app --reload
```

---
2024 클라우드시스템 팀 프로젝트
