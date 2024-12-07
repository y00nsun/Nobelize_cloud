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

dimension = 128  # 임베딩 차원
index = faiss.IndexFlatL2(dimension)
faiss.write_index(index, "embeddings/faiss_index_re.index")
