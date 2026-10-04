"""
Smart Study Notes Generator - Vercel serverless function (Python, stdlib only).

POST /api/summarize   body: {"text": "..."}
Calls the Hugging Face Inference API for facebook/bart-large-cnn, then adds
word counts, reduction % and key points (same logic as the desktop version).
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import urllib.error
import urllib.request

MODEL_NAME = os.environ.get("HF_MODEL", "facebook/bart-large-cnn")
HF_URL = os.environ.get(
    "HF_URL", f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MIN_WORDS = 30
MAX_INPUT_WORDS = 700

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "which", "who", "can", "will",
    "has", "have", "had", "not", "so", "than", "then", "they", "their", "we",
    "our", "you", "your", "also", "into", "such", "more", "most", "other",
}


# ---------------- Core logic ----------------
def count_words(text):
    return len(text.split())


def calculate_reduction(original_count, summary_count):
    if original_count == 0:
        return 0.0
    return ((original_count - summary_count) / original_count) * 100


def get_key_points(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]
    if not sentences:
        return []

    frequency = {}
    for word in re.findall(r"[a-zA-Z']+", text.lower()):
        if word not in STOP_WORDS and len(word) > 2:
            frequency[word] = frequency.get(word, 0) + 1

    scores = []
    for index, sentence in enumerate(sentences):
        words = re.findall(r"[a-zA-Z']+", sentence.lower())
        if not words:
            continue
        total = sum(frequency.get(w, 0) for w in words)
        scores.append((total / len(words), index))

    number_of_points = min(5, max(3, len(sentences) // 3), len(scores))
    best = sorted(scores, reverse=True)[:number_of_points]
    best_indexes = sorted(index for _, index in best)
    return [sentences[i] for i in best_indexes]


class SummaryError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def _post_to_hf(payload):
    request = urllib.request.Request(
        HF_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=55) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_summary(text, word_count):
    if not HF_TOKEN:
        raise SummaryError(
            "Server is missing the HF_TOKEN environment variable.", 500
        )

    words = text.split()
    if len(words) > MAX_INPUT_WORDS:
        text = " ".join(words[:MAX_INPUT_WORDS])

    max_length = min(130, max(25, int(word_count * 0.6)))
    min_length = max(10, max_length // 3)

    with_params = {
        "inputs": text,
        "parameters": {
            "generate_parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            }
        },
        "options": {"wait_for_model": True},
    }
    plain = {"inputs": text, "options": {"wait_for_model": True}}

    try:
        try:
            result = _post_to_hf(with_params)
        except urllib.error.HTTPError as error:
            # Some model/provider versions reject the length parameters.
            if error.code in (400, 422):
                result = _post_to_hf(plain)
            else:
                raise
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")[:300]
        if error.code in (401, 403):
            raise SummaryError("Hugging Face rejected the token (HF_TOKEN).", 500)
        if error.code == 503:
            raise SummaryError("The model is warming up. Try again in a minute.", 503)
        raise SummaryError(f"Hugging Face error {error.code}: {detail}")
    except urllib.error.URLError as error:
        raise SummaryError(f"Could not reach Hugging Face: {error.reason}")

    try:
        if isinstance(result, list):
            return result[0]["summary_text"].strip()
        return result["summary_text"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise SummaryError("Unexpected response from the summarization model.")


def build_response(raw_text):
    text = " ".join(raw_text.split())
    if not text:
        raise SummaryError("Please enter or paste a paragraph first.", 400)

    original_count = count_words(text)
    if original_count < MIN_WORDS:
        raise SummaryError(
            f"Your text has only {original_count} words. "
            f"Please enter at least {MIN_WORDS} words.",
            400,
        )

    summary = generate_summary(text, original_count)
    summary_count = count_words(summary)

    # Reduction is measured against the words the model actually read.
    used_count = min(original_count, MAX_INPUT_WORDS)
    return {
        "original_count": original_count,
        "summary": summary,
        "summary_count": summary_count,
        "reduction": round(calculate_reduction(used_count, summary_count), 2),
        "key_points": get_key_points(text),
        "truncated": original_count > MAX_INPUT_WORDS,
        "used_words": used_count,
    }


# ---------------- Vercel handler ----------------
class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 200_000:
                raise SummaryError("Text is too large (max ~200 KB).", 413)
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = build_response(str(payload.get("text", "")))
            self._send(200, result)
        except SummaryError as error:
            self._send(error.status, {"error": str(error)})
        except json.JSONDecodeError:
            self._send(400, {"error": "Invalid request body."})
        except Exception as error:
            self._send(500, {"error": f"Unexpected error: {error}"})

    def do_GET(self):
        self._send(200, {"status": "ok", "model": MODEL_NAME})
