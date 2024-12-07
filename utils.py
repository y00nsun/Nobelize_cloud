import openai
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os


def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트 추출"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def generate_embedding(text):
    """OpenAI API로 텍스트 임베딩 생성"""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])


def load_faiss_index(index_path, dimension):
    """FAISS 인덱스 로드 또는 초기화"""
    if os.path.exists(index_path):
        print(f"FAISS 인덱스를 {index_path}에서 로드합니다.")
        return faiss.read_index(index_path)
    else:
        print(f"{index_path}에 인덱스가 없습니다. 새 인덱스를 생성합니다.")
        return faiss.IndexFlatL2(dimension)


def save_faiss_index(index, index_path):
    """FAISS 인덱스를 파일로 저장"""
    faiss.write_index(index, index_path)
    print(f"FAISS 인덱스가 {index_path}에 저장되었습니다.")


def search_similar(embedding, faiss_index, k=5):
    """FAISS에서 유사 작품 검색"""
    distances, indices = faiss_index.search(np.array([embedding]), k)
    return indices, distances


def generate_feedback(text):
    """OpenAI API로 작품 평가 및 개선 방향 생성"""
    prompt = f"""
    아래의 문학 작품을 평가하고 개선 방향을 제시해주세요:

    작품 내용:
    {text}

    1. 문학적 스타일 평가:
    2. 개선 방향:
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response['choices'][0]['text'].strip()
