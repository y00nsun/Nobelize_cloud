from fastapi import FastAPI, UploadFile
import os
import numpy as np
from utils import extract_text_from_pdf, generate_embedding, load_faiss_index, save_faiss_index, search_similar, generate_feedback
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

# FastAPI 앱 생성
app = FastAPI()

# FAISS 인덱스 초기화
INDEX_PATH = "embeddings/faiss_index3.index"
DIMENSION = 1536
faiss_index = load_faiss_index(INDEX_PATH, DIMENSION)


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """PDF 업로드 및 FAISS 인덱스에 추가"""
    # PDF 저장
    pdf_path = f"data/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # 텍스트 추출 및 임베딩 생성
    text = extract_text_from_pdf(pdf_path)
    embedding = generate_embedding(text)

    # FAISS 인덱스에 추가
    faiss_index.add(np.array([embedding]))
    save_faiss_index(faiss_index, INDEX_PATH)

    return {"message": f"{file.filename} 업로드 및 인덱스에 추가됨"}


@app.post("/search")
async def search_similar_text(query: str):
    """입력 텍스트와 유사한 벡터 검색"""
    embedding = generate_embedding(query)
    indices, distances = search_similar(embedding, faiss_index)

    return {"indices": indices.tolist(), "distances": distances.tolist()}


@app.post("/evaluate")
async def evaluate_text(query: str):
    """입력 텍스트 평가 및 개선 방향 생성"""
    feedback = generate_feedback(query)
    return {"feedback": feedback}
