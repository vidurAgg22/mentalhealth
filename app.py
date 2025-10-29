# app.py
from flask import Flask, request, jsonify, render_template, session
import os
from rag_model import save_embeddings, answer_query

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# Ensure embeddings are prepared at startup
try:
    save_embeddings()
    print("Embeddings saved/updated.")
except Exception as e:
    print("Error preparing embeddings:", e)

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "hurt myself", "harm myself", "i'm going to kill"
]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    lower = user_input.lower()
    if any(keyword in lower for keyword in CRISIS_KEYWORDS):
        # Crisis response (not a substitute for professional help)
        crisis_msg = (
            "I'm sorry you're feeling this way. If you're in immediate danger, please call your local emergency number (for example 911 or 112). "
            "If you're in the United States, you can call or text 988 to reach the Suicide & Crisis Lifeline. "
            "If you are outside the US, please contact local emergency services or a trusted person nearby. "
            "Would you like resources for coping strategies or to find professional help?"
        )
        return jsonify({"response": crisis_msg, "is_crisis": True})

    # Regular knowledge-base lookup
    results = answer_query(user_input, top_k=1)
    if not results:
        return jsonify({"response": "Sorry, I couldn't find an answer. Try rephrasing or ask another question."})
    best = results[0]
    reply = f"{best['answer']} (confidence: {best['score']:.2f})"
    return jsonify({"response": reply, "source_question": best.get("question"), "score": best.get("score")})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
