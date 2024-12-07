import requests

# FastAPI 서버 URL
BASE_URL = "http://127.0.0.1:8000"

# 테스트용 파일 경로
TEST_FILE_PATH = "data/hyplex.pdf"

# 1. PDF 업로드 테스트
def test_upload():
    url = f"{BASE_URL}/upload"
    try:
        # 파일 열기 및 업로드 요청
        with open(TEST_FILE_PATH, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, files=files)
        
        # 응답 확인
        if response.status_code == 200:
            print("Upload Response:", response.json())
        else:
            print(f"Upload Failed. Status Code: {response.status_code}, Response: {response.text}")
    except FileNotFoundError:
        print(f"File not found: {TEST_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")


# 2. 검색 테스트
def test_search():
    url = f"{BASE_URL}/search"
    data = {"query": "example text"}
    try:
        # 검색 요청
        response = requests.post(url, json=data)
        
        # 응답 확인
        if response.status_code == 200:
            print("Search Response:", response.json())
        else:
            print(f"Search Failed. Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"An error occurred during search: {e}")


if __name__ == "__main__":
    print("Testing PDF Upload:")
    test_upload()

    print("\nTesting Search:")
    test_search()
