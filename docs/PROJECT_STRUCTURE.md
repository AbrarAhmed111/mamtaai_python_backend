## Project structure and file purposes

This backend is a minimal FastAPI template made to expose Python/AI utilities to a Next.js frontend. Below is a concise, file‑by‑file guide explaining what each part does and when you’d edit it.

### How a request flows
- Client calls an endpoint (for example, POST `/api/ai/wordcount`).
- The route handler (in `api/routers/ai.py`) validates the input using Pydantic models.
- The handler calls reusable logic in `services/` (pure Python functions).
- A JSON response is returned to the client.

---

### Top-level files
- `README.md`: Quickstart and usage. Start here for install, run, and endpoints.
- `requirements.txt`: Python dependencies. Keep minimal; add only what you need.
- `env.example`: Sample environment variables. Copy to `.env` and adjust values.
- `config.py`: Central place to read environment variables (e.g., `ALLOWED_ORIGINS`, `ENVIRONMENT`) and expose a simple config object if needed.
- `Procfile`: Process definition for Railway or similar PaaS. Runs the app with uvicorn.
- `railway.json`: Railway deployment settings (builder type, restart policy).
- `runtime.txt`: Python version pin (used by certain platforms).
- `start_api_server.py`: Optional local runner from project root using uvicorn. Handy if you prefer `python start_api_server.py` over typing the uvicorn command.

---

### Application package: `api/`
- `api/__init__.py`: Marks `api` as a Python package; stores a simple version string.
- `api/main.py`: The main FastAPI application.
  - Creates the `FastAPI` app.
  - Configures CORS (so your Next.js app can call it).
  - Mounts routers under `/api` (for example, `ai` routes).
  - Defines basic utility endpoints like `/` and `/health`.
  - When adding new feature areas, import and include new routers here.

#### Routers
- `api/routers/ai.py`:
  - Example router demonstrating common patterns:
    - Input validation with Pydantic (`FibonacciRequest`, `WordCountRequest`, etc.).
    - JSON responses with Pydantic response models.
  - Endpoints:
    - `POST /api/ai/fibonacci` → compute first N Fibonacci numbers.
    - `POST /api/ai/wordcount` → word frequency in text.
    - `POST /api/ai/normalize` → min–max normalization for arrays of numbers.
  - Use this file as a template to build your own routers (e.g., `vision.py`, `nlp.py`).

---

### Service layer: `services/`
- Purpose: Pure Python logic (no web framework code). Easy to test and reuse.
- `services/__init__.py`: Marks `services` as a package.
- `services/ai.py`: Example utility functions used by the `ai` router:
  - `generate_fibonacci(count)`, `count_words(text)`, `normalize_numbers(values)`.
  - Add your own domain logic here (AI/ML processing, data transforms, etc.).

---

### Environment and configuration
- Create a `.env` file by copying `env.example`:
  - `ENVIRONMENT` (e.g., development or production)
  - `BASE_URL` (e.g., `http://localhost:8000`)
  - `API_VERSION` (semantic version for your service)
  - `ALLOWED_ORIGINS` (e.g., `*` or `http://localhost:3000` for Next.js)
- `config.py` reads those values and provides helpers; import it anywhere you need configuration.

---

### Common tasks
1) Add a new endpoint
   - Create a new router file in `api/routers/` (e.g., `vision.py`).
   - Define Pydantic request/response models and route handlers.
   - In `api/main.py`, import your router and include it with `app.include_router(...)`.

2) Add new business logic
   - Put pure functions in `services/` (e.g., `services/vision.py`).
   - Call those functions from your router handlers.

3) Enable CORS for Next.js during local dev
   - Set `ALLOWED_ORIGINS=http://localhost:3000` in `.env`.
   - Restart the server.

4) Run locally
   - `pip install -r requirements.txt`
   - `uvicorn api.main:app --reload`
   - or `python start_api_server.py`

5) Deploy (Railway)
   - Link your repo to Railway.
   - Ensure environment variables are set in Railway.
   - `Procfile` already specifies the start command.

---

### Tips
- Keep routers focused on HTTP concerns (validation, response shaping).
- Keep `services/` focused on computation/logic; it should not know about HTTP.
- Use Pydantic models to validate and document request/response payloads.
- Keep dependencies minimal; add heavy libraries only when needed.

