# ✨✍️Nobelize

**AI-powered writing evaluation and refinement service — “Nobelize.”**

- “Nobel (Nobel Prize in Literature) + Realize (to bring into reality)”
- A writing enhancement system built using AI models and a Docker/Kubernetes-based cloud infrastructure.

---

## Project Structure

```plaintext
Nobelize_cloud/
├── main.py                  # FastAPI server entry point
├── utils.py                 # FAISS and OpenAI utility functions
├── embeddings/              # Stored FAISS index
│   └── faiss_index3.index
├── data/                    # PDF storage directory
│   └── example.pdf          # Sample test PDF
├── requirements.txt         # Package dependencies
├── .env                     # Environment variables (OpenAI API Key)
└── .gitignore               # Files to ignore (e.g., .env)
```

---

## 1. Python Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   **Windows (CMD):**
   ```bash
   venv\Scripts\activate
   ```

   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\activate
   ```

   **Linux / macOS:**
   ```bash
   source venv/bin/activate
   ```

3. Deactivate:
   ```bash
   deactivate
   ```


## 2. Install Dependencies

```bash
pip install -r requirements.txt
```


## 3. Run the Server

```bash
.\venv\Scripts\activate
uvicorn main:app --reload
```

---

2024 Cloud System Team Project
