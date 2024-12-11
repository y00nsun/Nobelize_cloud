# Python 3.9 Slim 이미지를 기반으로 사용
FROM python:3.9-slim

# 컨테이너 내 작업 디렉터리 설정
WORKDIR /app

# 의존성 설치를 위해 requirements.txt 복사
COPY requirements.txt .

# NumPy 설치 후 다른 의존성 설치
RUN pip install --no-cache-dir "numpy<2" && pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
