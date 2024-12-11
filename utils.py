import openai
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
from openai import OpenAI
client = OpenAI()


def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트 추출"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def generate_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


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
    # FAISS 검색
    distances, indices = faiss_index.search(np.array([embedding]).astype('float32'), k)

    # indices와 distances 반환 (1D로 압축)
    return indices[0], distances[0]


# def generate_feedback(text):
#     """OpenAI API로 작품 평가 및 개선 방향 생성"""
#     prompt = f"""
#     아래의 문학 작품을 평가하고 개선 방향을 제시해주세요:

#     작품 내용:
#     {text}

#     1. 문학적 스타일 평가:
#     2. 개선 방향:
#     """
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt,
#         max_tokens=500
#     )
#     return response['choices'][0]['text'].strip()

# def generate_feedback(user_text, closest_text, query):
#     """OpenAI API로 작품 평가 및 개선 방향 생성"""
#     prompt = f"""
#     사용자의 문학 작품을 평가하고 개선 방향을 제시해주세요:
    
#     사용자의 작품 내용:
#     {user_text}
    
#     노벨 수상작과 비교 (가장 유사한 작품 내용):
#     {closest_text}
    
#     사용자 질문:
#     {query}

#     1. 사용자 작품의 문학적 스타일 평가:
#     2. 노벨 수상작과의 주요 차이점:
#     3. 개선 방향 제시:
#     """
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt,
#         max_tokens=500
#     )
#     return response['choices'][0]['text'].strip()

# def generate_feedback(query):
#    # 4. OpenAI GPT를 사용해 응답 생성
#     prompt = f"사용자의 문학 작품을 평가하고 개선 방향을 제시해주세요:
    
#     사용자의 작품 내용:
#     {user_text}
    
#     노벨 수상작과 비교 (가장 유사한 작품 내용):
#     {closest_text}
    
#     사용자 질문:
#     {query}

#     1. 사용자 작품의 문학적 스타일 평가:
#     2. 노벨 수상작과의 주요 차이점:
#     3. 개선 방향 제시:
#     "
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "당신은 문맥 기반 정보를 활용해 사용자 질문에 정확하고 친절하게 답변하는 한국어 전문가입니다."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
#     answer = completion.choices[0].message

#     return {"answer": answer}

def generate_feedback(user_text, closest_text,query):
   # 4. OpenAI GPT를 사용해 응답 생성
    prompt = f"""사용자의 문학 작품을 평가하고 개선 방향을 제시해주세요:
    
    사용자의 작품 내용:
    {user_text}
    
    노벨 수상작과 비교 (가장 유사한 작품 내용):
    {closest_text}
    
    사용자 질문:
    {query}

    1. 사용자 작품의 문학적 스타일 평가:
    2. 노벨 수상작과의 주요 차이점:
    3. 개선 방향 제시:
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "당신은 문맥 기반 정보를 활용해 사용자 질문에 정확하고 친절하게 답변하는 한국어 전문가입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer= completion.choices[0].message.content

    return {"answer": answer}



