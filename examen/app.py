"""Servicio de recuperacion lexica para SciFact (TF-IDF y BM25)."""
from __future__ import annotations

import html
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import pandas as pd
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", BASE_DIR / "data/scifact/corpus/corpus.parquet"))

# Descarga al iniciar para que los workers de Render tengan el recurso disponible.
nltk.download("stopwords", quiet=True)
try:
    STOP = set(stopwords.words("english"))
except LookupError as exc:
    raise RuntimeError("No se pudo cargar la lista stopwords de NLTK.") from exc
STEM = PorterStemmer()
app = Flask(__name__)


def preprocesar(texto: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", str(texto).lower())
    return [STEM.stem(t) for t in tokens if t not in STOP and len(t) > 2]


def cargar_corpus(ruta: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el corpus: {ruta}")
    df = pd.read_parquet(ruta)
    required = {"_id", "title", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"El corpus debe contener las columnas: {sorted(required)}")
    docs, texts = {}, {}
    for doc_id, title, text in zip(df["_id"], df["title"], df["text"]):
        title = html.unescape(str(title or "")).strip()
        body = html.unescape(str(text or "")).replace('""', '"').strip()
        doc_id = str(doc_id)
        docs[doc_id] = {"title": title, "text": body}
        texts[doc_id] = f"{title} {body}".strip()
    return docs, texts


def construir_indice(texts: dict[str, str]) -> dict:
    indice, longitudes = defaultdict(dict), {}
    for doc_id, texto in texts.items():
        tokens = preprocesar(texto)
        longitudes[doc_id] = len(tokens)
        for token, frecuencia in Counter(tokens).items():
            indice[token][doc_id] = frecuencia
    n = len(texts)
    if not n:
        raise ValueError("El corpus no contiene documentos.")
    idx = {"indice": dict(indice), "len": longitudes, "N": n,
           "avgdl": sum(longitudes.values()) / n}
    normas = defaultdict(float)
    for token, postings in idx["indice"].items():
        peso_idf = math.log(1 + n / len(postings))
        for doc_id, freq in postings.items():
            normas[doc_id] += ((1 + math.log(freq)) * peso_idf) ** 2
    idx["normas"] = {d: math.sqrt(v) for d, v in normas.items()}
    return idx


def tfidf(consulta: str, idx: dict) -> dict[str, float]:
    q = Counter(preprocesar(consulta))
    scores, norma_q = defaultdict(float), 0.0
    for token, fq in q.items():
        postings = idx["indice"].get(token)
        if not postings:
            continue
        idf = math.log(1 + idx["N"] / len(postings))
        wq = (1 + math.log(fq)) * idf
        norma_q += wq ** 2
        for doc_id, freq in postings.items():
            scores[doc_id] += wq * (1 + math.log(freq)) * idf
    norma_q = math.sqrt(norma_q) or 1.0
    return {d: score / (norma_q * idx["normas"][d])
            for d, score in scores.items() if idx["normas"].get(d, 0)}


def bm25(consulta: str, idx: dict, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    scores = defaultdict(float)
    for token in set(preprocesar(consulta)):
        postings = idx["indice"].get(token)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (idx["N"] - df + 0.5) / (df + 0.5))
        for doc_id, freq in postings.items():
            norm = freq + k1 * (1 - b + b * idx["len"][doc_id] / idx["avgdl"])
            scores[doc_id] += idf * freq * (k1 + 1) / norm
    return dict(scores)


def buscar(modelo: str, consulta: str, top_k: int) -> list[tuple[str, float]]:
    scores = bm25(consulta, idx) if modelo == "bm25" else tfidf(consulta, idx)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def parsear_top_k(raw, desde_url: bool):
    """Devuelve un entero entre 1 y 100, o None si no es valido.

    En GET los parametros llegan como texto ("5"), en POST/JSON como numero.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str) and desde_url:
        raw = raw.strip()
        if not raw.isdigit():
            return None
        raw = int(raw)
    if not isinstance(raw, int) or not 1 <= raw <= 100:
        return None
    return raw


print(f"Cargando corpus desde {CORPUS_PATH} ...")
docs, texts = cargar_corpus(CORPUS_PATH)
idx = construir_indice(texts)
print(f"Servicio listo: {idx['N']} documentos indexados.")


@app.get("/")
def inicio():
    return jsonify({
        "service": "Buscador SciFact (TF-IDF y BM25)",
        "endpoints": {
            "GET /health": "estado del servicio",
            "GET /search?query=...&model=bm25|tfidf&top_k=10": "busqueda por URL",
            "POST /search": "busqueda con JSON {query, model, top_k}",
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "message": "Servicio de busqueda activo",
                    "documents_indexed": idx["N"]})


@app.route("/search", methods=["GET", "POST"])
def search():
    # GET: parametros en la URL. POST: cuerpo JSON.
    desde_url = request.method == "GET"
    if desde_url:
        data = request.args.to_dict()
    else:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json."}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "El cuerpo debe ser un objeto JSON valido."}), 400

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "'query' debe ser un texto no vacio."}), 400

    model = data.get("model", "bm25")
    if not isinstance(model, str) or model.lower() not in {"bm25", "tfidf"}:
        return jsonify({"error": "'model' debe ser 'bm25' o 'tfidf'."}), 400
    model = model.lower()

    top_k = parsear_top_k(data.get("top_k", 10), desde_url)
    if top_k is None:
        return jsonify({"error": "'top_k' debe ser un entero entre 1 y 100."}), 400

    ranking = buscar(model, query, top_k)
    results = []
    for rank, (doc_id, score) in enumerate(ranking, 1):
        doc = docs[doc_id]
        snippet_source = doc["text"] or doc["title"]
        snippet = snippet_source[:240] + ("..." if len(snippet_source) > 240 else "")
        results.append({"rank": rank, "document_id": doc_id, "score": round(float(score), 6),
                        "title": doc["title"], "snippet": snippet})
    return jsonify({"model": model, "query": query, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)