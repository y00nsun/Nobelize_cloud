
---
### **1. Python 가상환경 생성**
1. **가상환경 생성**:
   프로젝트 디렉터리에서 다음 명령어를 실행하여 가상환경을 생성합니다:

   ```bash
   python -m venv venv
   ```

2. **가상환경 활성화**:
   - **Windows (CMD)**:
     ```bash
     venv\Scripts\activate
     ```

   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\activate
     ```

   - **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```

3. **가상환경 비활성화**:
   ```bash
   deactivate
   ```

---

### **2. `requirements.txt`**
```bash
pip install -r requirements.txt
```