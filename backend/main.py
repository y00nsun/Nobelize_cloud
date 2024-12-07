from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import numpy as np
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 OpenAI API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# FastAPI 앱 생성
app = FastAPI()

# 요청 데이터 모델
class TextInput(BaseModel):
    text: str

@app.post("/evaluate")
async def evaluate_text(input: TextInput):
    try:
        # OpenAI API를 사용하여 텍스트 임베딩 생성
        response = openai.Embedding.create(
            input=input.text,
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'])
        return {"embedding": embedding.tolist()}
    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
