from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 도메인으로 제한하려면 ["http://example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FAISS 인덱스 초기화
INDEX_PATH = "embeddings/faiss_index3.index"
DIMENSION = 1536
faiss_index = load_faiss_index(INDEX_PATH, DIMENSION)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("/paper.ico")

# Static 파일 경로 설정
app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("./static/frontend.html")

#upload type1: 사용자가 자기 작품을 업로드 -> data 아래에 저장된다.
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """PDF 업로드 및 FAISS 인덱스에 추가"""
    # PDF 저장
    pdf_path = f"data/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    return {"message": f"{file.filename} 업로드 및 인덱스에 추가됨"}


@app.post("/search")
async def search_similar_text(query: str):
    """입력 텍스트와 유사한 벡터 검색"""
    embedding = generate_embedding(query)
    indices, distances = search_similar(embedding, faiss_index)

    return {"indices": indices.tolist(), "distances": distances.tolist()}


@app.post("/evaluate")
async def evaluate_text(filename:str, query: str):
    """입력 텍스트 평가 및 개선 방향 생성"""

    # 기존의 search api 내부에 있던 것
    #  텍스트 추출 및 임베딩 생성
    pdf_path = f"data/{filename}"
    text = extract_text_from_pdf(pdf_path)
    user_embedding = generate_embedding(text)

    indices, distances = search_similar(user_embedding, faiss_index)
    # 가장 가까운 인덱스와 거리
    closest_index = int(indices[0])  # 1D 배열의 첫 번째 값 추출
    closest_text = faiss_index.reconstruct(closest_index)  # FAISS에서 벡터 복원
    
    feedback = generate_feedback(text, closest_text, query)
    return {"feedback": feedback}