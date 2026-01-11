from flask import Flask, render_template, request, jsonify, Response
from data_loader import load_sarcasm_dataset
from evaluator import classify_sentence, run_batch_stream
import json

app = Flask(__name__)

DATASET_PATH = "data/Sarcasm_Headlines_Dataset.json"
BATCH_SIZE = 10
INTERACTIVE_SIZE = 20
API_DELAY = 5


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_data", methods=["GET"])
def load_data():
    _, interactive_records = load_sarcasm_dataset(
        DATASET_PATH, BATCH_SIZE, INTERACTIVE_SIZE
    )
    return jsonify(interactive_records)


@app.route("/single_test", methods=["POST"])
def single_test():
    data = request.json
    sentence = data["sentence"]
    actual_label = data["actual_label"]
    gemini_key = data["gemini_key"]
    groq_key = data["groq_key"]

    gemini_pred, groq_pred = classify_sentence(sentence, gemini_key, groq_key)

    return jsonify({
        "sentence": sentence,
        "actual": actual_label,
        "gemini": gemini_pred,
        "groq": groq_pred
    })


@app.route("/batch_test_stream", methods=["GET"])
def batch_test_stream():
    gemini_key = request.args.get("gemini_key")
    groq_key = request.args.get("groq_key")

    batch_records, _ = load_sarcasm_dataset(
        DATASET_PATH, BATCH_SIZE, INTERACTIVE_SIZE
    )

    def generate():
        for result in run_batch_stream(batch_records, gemini_key, groq_key, API_DELAY):
            yield f"data: {json.dumps(result)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
