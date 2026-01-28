# CVision

Simple web app to upload a PDF CV and get feedback on weaknesses, improvements, and an overall score.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` and upload a PDF.

## Troubleshooting

If you see an error like `Directory '.../app/static' does not exist`, make sure the
`app/static` folder exists (it is included in the repo). This directory is required
by the FastAPI static file mount in `app/main.py`.

## Configuration

- `CVISION_ANALYZER` (default: `heuristic`): choose the analysis backend. Supports `heuristic` and `ollama`.
- `CVISION_OLLAMA_URL` (default: `http://localhost:11434`): base URL for a local Ollama instance.
- `CVISION_OLLAMA_MODEL` (default: `llama3.1`): model name to use with Ollama.

### Free AI option (local Ollama)

If you want AI-based analysis without paid APIs, run [Ollama](https://ollama.com/) locally and set:

```bash
export CVISION_ANALYZER=ollama
export CVISION_OLLAMA_MODEL=llama3.1
```

The app will attempt the AI analysis first and fall back to the heuristic analyzer if the response
is invalid or unavailable.
