from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import openai
import faiss
import os
import numpy as np
from utils import extract_text_from_pdf, generate_embedding, load_faiss_index

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 인스턴스 생성
app = FastAPI()

# FAISS 인덱스 초기화
INDEX_PATH = "embeddings/faiss_index"
DIMENSION = 1536
faiss_index = load_faiss_index(INDEX_PATH, DIMENSION)

# PDF 업로드 및 FAISS 인덱스 업데이트
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    pdf_path = f"data/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    text = extract_text_from_pdf(pdf_path)
    embedding = generate_embedding(text)
    faiss_index.add(np.array([embedding]))
    faiss.write_index(faiss_index, INDEX_PATH)
    return {"message": f"{file.filename} 업로드 및 인덱스에 추가됨"}

# 텍스트 검색
@app.post("/search")
async def search_similar(query: str):
    embedding = generate_embedding(query)
    distances, indices = faiss_index.search(np.array([embedding]), k=5)
    return {"indices": indices.tolist(), "distances": distances.tolist()}
