from __future__ import annotations

import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader

app = FastAPI(title="CVision")

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@dataclass
class CvAnalysis:
    weaknesses: List[str]
    improvements: List[str]
    score: int


class CvAnalyzer:
    def analyze(self, text: str) -> CvAnalysis:
        raise NotImplementedError


class HeuristicCvAnalyzer(CvAnalyzer):
    def analyze(self, text: str) -> CvAnalysis:
        lowered = text.lower()
        words = re.findall(r"\b\w+\b", lowered)
        word_count = len(words)

        weaknesses: List[str] = []
        improvements: List[str] = []
        score = 100

        if not re.search(r"\b[\w.-]+@[\w.-]+\.[a-z]{2,}\b", lowered):
            weaknesses.append("No email address detected.")
            improvements.append("Add a professional email address near your contact details.")
            score -= 12

        if not re.search(r"\b(\+?\d[\d\s\-()]{7,})\b", lowered):
            weaknesses.append("No phone number detected.")
            improvements.append("Include a reachable phone number.")
            score -= 8

        if "linkedin" not in lowered:
            weaknesses.append("No LinkedIn profile detected.")
            improvements.append("Add your LinkedIn URL if it is updated and relevant.")
            score -= 6

        if word_count < 250:
            weaknesses.append("The CV looks quite short.")
            improvements.append("Add more detail on impact, achievements, and responsibilities.")
            score -= 10
        elif word_count > 900:
            weaknesses.append("The CV may be too long.")
            improvements.append("Trim less relevant content and keep it concise.")
            score -= 8

        if "experience" not in lowered and "work history" not in lowered:
            weaknesses.append("Work experience section not detected.")
            improvements.append("Add a dedicated Experience section with roles and achievements.")
            score -= 12

        if "education" not in lowered:
            weaknesses.append("Education section not detected.")
            improvements.append("Add your education background and relevant certifications.")
            score -= 8

        if "skills" not in lowered:
            weaknesses.append("Skills section not detected.")
            improvements.append("Include a Skills section tailored to the roles you want.")
            score -= 8

        if re.search(r"\b(lorem ipsum|dummy)\b", lowered):
            weaknesses.append("Placeholder text detected.")
            improvements.append("Replace placeholder text with real accomplishments.")
            score -= 10

        if not weaknesses:
            weaknesses.append("No major structural issues detected.")
            improvements.append("Consider refining formatting and adding measurable outcomes.")

        score = max(0, min(100, score))
        return CvAnalysis(weaknesses=weaknesses, improvements=improvements, score=score)


class OllamaCvAnalyzer(CvAnalyzer):
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def analyze(self, text: str) -> CvAnalysis:
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(text),
            "stream": False,
        }
        response = self._post_json(f"{self.base_url}/api/generate", payload)
        content = response.get("response", "")
        parsed = self._parse_response(content)
        if parsed:
            return parsed
        return HeuristicCvAnalyzer().analyze(text)

    @staticmethod
    def _build_prompt(text: str) -> str:
        return (
            "You are an ATS-style CV reviewer. Analyze the CV text and respond ONLY with JSON "
            "matching this schema: "
            '{"weaknesses":[string], "improvements":[string], "score": integer 0-100}. '
            "Keep weaknesses and improvements concise and actionable.\n\n"
            f"CV TEXT:\n{text}"
        )

    @staticmethod
    def _post_json(url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _parse_response(content: str) -> CvAnalysis | None:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        weaknesses = data.get("weaknesses")
        improvements = data.get("improvements")
        score = data.get("score")

        if not isinstance(weaknesses, list) or not isinstance(improvements, list):
            return None
        if not isinstance(score, int):
            return None

        cleaned_weaknesses = [str(item).strip() for item in weaknesses if str(item).strip()]
        cleaned_improvements = [str(item).strip() for item in improvements if str(item).strip()]
        bounded_score = max(0, min(100, score))

        if not cleaned_weaknesses or not cleaned_improvements:
            return None

        return CvAnalysis(
            weaknesses=cleaned_weaknesses,
            improvements=cleaned_improvements,
            score=bounded_score,
        )


def get_analyzer() -> CvAnalyzer:
    backend = os.getenv("CVISION_ANALYZER", "heuristic").lower()
    if backend == "heuristic":
        return HeuristicCvAnalyzer()
    if backend == "ollama":
        model = os.getenv("CVISION_OLLAMA_MODEL", "llama3.1")
        base_url = os.getenv("CVISION_OLLAMA_URL", "http://localhost:11434")
        return OllamaCvAnalyzer(model=model, base_url=base_url)
    return HeuristicCvAnalyzer()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, cv_file: UploadFile = File(...)):
    if cv_file.content_type != "application/pdf":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Please upload a PDF file.",
            },
        )

    file_bytes = await cv_file.read()
    if not file_bytes:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Uploaded file is empty.",
            },
        )

    text = extract_text_from_pdf(file_bytes)
    if not text:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Could not extract text from the PDF.",
            },
        )

    analyzer = get_analyzer()
    analysis = analyzer.analyze(text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "analysis": analysis,
            "word_count": len(re.findall(r"\b\w+\b", text)),
        },
    )
