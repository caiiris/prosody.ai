"""
Poetry era classifier webapp.
Paste a poem → get closest time period match.
Backend uses placeholder response until model pipeline is trained and saved.
"""
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "poem" not in data:
        return jsonify({"error": "Missing 'poem' in request body"}), 400

    poem = data["poem"].strip()
    if not poem:
        return jsonify({"error": "Poem text is empty"}), 400

    # Placeholder: return mock result until real model is wired up.
    # Later: load model + feature pipeline, run feature extraction, predict era.
    result = _placeholder_predict(poem)
    return jsonify(result)


def _placeholder_predict(poem: str) -> dict:
    """Return mock era prediction. Replace with real model inference later."""
    word_count = len(poem.split())
    # Simple heuristic for demo: very short = "need more text", else mock era
    if word_count < 10:
        return {
            "era": None,
            "confidence": 0.0,
            "alternatives": [],
            "message": "Please paste a longer poem (at least a few lines) for analysis.",
        }
    # Mock response — swap this with loading model and calling model.predict()
    return {
        "era": "Contemporary",
        "confidence": 0.72,
        "alternatives": [
            {"era": "Mid-20th century", "confidence": 0.18},
            {"era": "Modernist", "confidence": 0.10},
        ],
        "message": None,
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
