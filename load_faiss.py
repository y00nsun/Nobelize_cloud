# faiss index 파일 손상 여부 확인용

# import faiss
# import os

# index_path = "embeddings/faiss_index3.index"

# if os.path.exists(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         print("FAISS 인덱스 로드 성공")
#     except Exception as e:
#         print(f"FAISS 인덱스 로드 실패: {e}")
# else:
#     print(f"인덱스 파일 {index_path}이 존재하지 않습니다.")

import faiss
import numpy as np
import os

# 1. 임베딩 벡터의 차원
dimension = 1536
index = faiss.IndexFlatL2(dimension)

# 2. 디렉토리 생성
index_path = "embeddings/faiss_index3.index"
os.makedirs(os.path.dirname(index_path), exist_ok=True)
