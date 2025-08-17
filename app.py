# app.py
from flask_cors import CORS
from flask import Flask, request, jsonify
import requests
import numpy as np
from generate_embedding import get_embedding_from_base64

app = Flask(__name__)
CORS(app)
FETCH_EMBEDDINGS_URL = "dummy url for copy"

@app.route("/match-face", methods=["POST", "OPTIONS"])
def match_face():
    if request.method == "OPTIONS":
        # Preflight request
        # response = app.make_default_options_response()
        # You can also add/modify headers here if needed:
        response.headers["Access-Control-Allow-Origin"] = "sdm-connect-2.netlify.app"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    data = request.get_json()
    username = data.get("username")
    image_base64 = data.get("imageBase64")

    if not username or not image_base64:
        return jsonify({"error": "Missing username or imageBase64"}), 400

    try:
        input_embedding = get_embedding_from_base64(image_base64)
    except Exception as e:
        return jsonify({"error": f"Failed to generate embedding: {str(e)}"}), 400

    # Fetch stored embeddings
    try:
        resp = requests.get(FETCH_EMBEDDINGS_URL, params={"username": username})
        resp.raise_for_status()
        stored_embeddings = resp.json().get("records", {})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stored embeddings: {str(e)}"}), 500

    best_match = None
    lowest_distance = float("inf")

    for student_name, vectors in stored_embeddings.items():
        for obj in vectors:
            stored_vec = obj["embedding"] if isinstance(obj, dict) else obj
            dist = np.linalg.norm(np.array(input_embedding) - np.array(stored_vec))
            if dist < lowest_distance:
                lowest_distance = dist
                best_match = student_name

    if best_match:
        return jsonify({"bestMatch": best_match, "distance": lowest_distance})
    else:
        return jsonify({"error": "No confident match found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
