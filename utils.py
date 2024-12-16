import openai
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")



def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트 추출"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def generate_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    MAX_TOKENS = 8192
     # 텍스트를 청크로 나누기
    def split_into_chunks(text, chunk_size):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # 청크 생성
    CHUNK_SIZE = MAX_TOKENS * 4  # 토큰당 약 4글자 기준으로 계산
    chunks = split_into_chunks(text, CHUNK_SIZE)

    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(input=chunk, model=model).data[0].embedding
        embeddings.append(response)

    # 여러 임베딩 평균화
    final_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
    return final_embedding
  
    # return np.array(response['data'][0]['embedding'])


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


def generate_feedback(user_text, closest_text,query):
    MAX_PROMPT_TOKENS = 8192  # 모델의 최대 토큰 제한
    SAFETY_MARGIN = 500  # completion 토큰을 위해 여유 공간 확보
     # 텍스트를 최대 길이에 맞게 자르기
    def truncate_text(text, max_tokens):
        return text[:max_tokens * 4]  # 대략적인 문자 길이 기준

    user_text_truncated = truncate_text(user_text, (MAX_PROMPT_TOKENS - SAFETY_MARGIN) // 2)
    closest_text_truncated = truncate_text(closest_text, (MAX_PROMPT_TOKENS - SAFETY_MARGIN) // 2)

   # 4. OpenAI GPT를 사용해 응답 생성
    prompt = f"""사용자의 문학 작품을 평가하고 개선 방향을 제시해주세요:
    
    사용자의 작품 내용:
    {user_text_truncated}
    
    노벨 수상작과 비교 (가장 유사한 작품 내용):
    {closest_text_truncated}
    
    사용자 질문:
    {query}

    1. 사용자 작품의 문학적 스타일 평가:
    2. 노벨 수상작과의 주요 차이점:
    3. 개선 방향 제시:
    """
    completion = openai.ChatCompletion.create(
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
    # return completion['choices'][0]['message']['content']

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

