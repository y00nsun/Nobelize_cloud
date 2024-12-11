# from utils import generate_embedding

# # 테스트 텍스트
# sample_text = "This is a sample text for embedding."

# # 임베딩 생성 및 차원 출력
# embedding = generate_embedding(sample_text)
# print(f"Embedding dimension: {embedding.shape[0]}")

import faiss
import numpy as np

# FAISS 인덱스 파일 경로
index_path = "embeddings/faiss_index3.index"

# FAISS 인덱스 로드
try:
    index = faiss.read_index(index_path)
    print(f"FAISS 인덱스가 성공적으로 로드되었습니다.")
    print(f"FAISS 인덱스의 차원: {index.d}")
    print(f"FAISS 인덱스에 저장된 벡터 수: {index.ntotal}")
except FileNotFoundError:
    print(f"FAISS 인덱스 파일 {index_path}이(가) 존재하지 않습니다.")
except Exception as e:
    print(f"오류 발생: {e}")




# old_index = faiss.read_index("embeddings/faiss_index3.index")
# old_data = old_index.reconstruct_n(0, old_index.ntotal)

# new_data = np.pad(old_data, ((0, 0), (0, 1536 - 384)), mode='constant')

# new_index = faiss.IndexFlatL2(1536)
# new_index.add(new_data)
# faiss.write_index(new_index, "embeddings/faiss_index3.index")
# print("FAISS 인덱스가 업데이트되었습니다.")


