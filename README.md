# PulseSense – AI-Powered Health Diagnostics

AI-assisted health companion that combines a FastAPI backend, MySQL persistence, and a Streamlit frontend. PulseSense lets users register, log in, manage medical records, run symptom checks powered by a lightweight RAG pipeline, and chat with a health-focused assistant. The backend initializes seed medical data on startup and exposes JWT-protected APIs; the frontend provides a polished multi-page experience.

## Features
- Secure auth (register/login) with JWT bearer tokens and bcrypt password hashing
- Personal medical record management (create, list, delete)
- Symptom checker with rule-based + RAG-style suggestions (disease, confidence, specialist, tests, precautions)
- Health chatbot that stores conversations per user
- Profile view and dashboard metrics in Streamlit
- Built-in database bootstrap (tables, sample symptoms/diseases, mappings)
- CORS enabled for local development

## Project Structure
- `backend/` – FastAPI service (`app/main.py`) with auth, RAG agent, and DB access
- `frontend/` – Streamlit UI (`app.py`) consuming the backend
- `database/` – SQL bootstrap scripts and helper runner

## Prerequisites
- Python 3.10+
- MySQL 8.x running locally (or reachable host)
- Node is **not** required (frontend is Streamlit)
- (Optional) GPU for faster transformer inference; CPU fallback is supported

## Quick Start
1) **Clone & enter**  
   ```bash
   git clone <repo-url>
   cd final dbms project
   ```

2) **Set up MySQL**  
   - Create a database and user, or use root:
   ```sql
   CREATE DATABASE pulsesense;
   ```
   - Option A: run the seed script once:
     ```bash
     python database/run_init_sql.py   # uses database/init.sql
     ```
   - Option B: rely on backend auto-bootstrap (creates tables, inserts sample symptoms/diseases on startup).

3) **Configure environment**  
   Create `backend/.env` (values shown are examples):
   ```
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=pulsesense
   SECRET_KEY=replace-with-strong-secret
   ```

4) **Backend setup & run**  
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   API docs: `http://localhost:8000/docs`

5) **Frontend setup & run**  
   ```bash
   cd frontend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
   The frontend expects the backend at `http://localhost:8000` (see `frontend/app.py` `BACKEND_URL`).

## API Overview (selected)
- `POST /register` – create user
- `POST /login` – obtain JWT (`access_token`)
- `GET /medical-records` – list current user records
- `POST /medical-records` – create record
- `DELETE /medical-records/{id}` – delete own record
- `GET /symptoms` – list seed symptoms
- `POST /predict-disease` – symptom-based suggestion (requires auth)
- `POST /chatbot/query` – health chatbot (stores conversation)
- `GET /rag-status` – diagnostics for RAG components

All protected routes require `Authorization: Bearer <token>` from `/login`.

## RAG / AI Notes
- Uses a lightweight knowledge base with sentence-transformer embeddings and FAISS when available; falls back to a simple in-memory vector store.
- Transformer models (e.g., `google/flan-t5-*`) download on first run; ensure internet or pre-cache.
- If transformers/langchain imports fail, the agent downgrades gracefully to a rule-based flow.

## Development Tips
- Backend auto-creates tables and inserts sample data at startup; disable if you manage schema manually.
- Update `SECRET_KEY` and DB credentials in production.
- Streamlit UI uses session state for navigation; `BACKEND_URL` can be pointed to a remote API.

## Troubleshooting
- **MySQL access denied**: verify `.env` credentials and that MySQL is running on the specified port.
- **Model download slow**: pre-download Hugging Face models or ensure a stable connection; GPU is optional.
- **CORS issues**: CORS is open for development; tighten in production within `backend/app/main.py`.

