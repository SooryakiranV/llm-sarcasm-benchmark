import time
import re
from llm_clients import call_gemini, call_groq

def normalize_label(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def classify_sentence(sentence, gemini_key, groq_key):
    prompt = f"Reply ONLY with 'Sarcastic' or 'Not Sarcastic'. Sentence: {sentence}"

    try:
        gemini_result = call_gemini(gemini_key, prompt)
    except Exception:
        gemini_result = "Unavailable"

    try:
        groq_result = call_groq(groq_key, prompt)
    except Exception:
        groq_result = "Unavailable"

    return gemini_result, groq_result


def run_batch_stream(records, gemini_key, groq_key, delay_seconds=5):
    gemini_correct = 0
    groq_correct = 0
    total = len(records)

    for i, item in enumerate(records):
        sentence = item["sentence"]
        actual = item["label"]

        gemini_pred, groq_pred = classify_sentence(sentence, gemini_key, groq_key)

        clean_actual = normalize_label(actual)
        clean_gemini = normalize_label(gemini_pred)
        clean_groq = normalize_label(groq_pred)

        gemini_ok = clean_gemini == clean_actual
        groq_ok = clean_groq == clean_actual

        if gemini_ok:
            gemini_correct += 1
        if groq_ok:
            groq_correct += 1

        yield {
            "index": i + 1,
            "total": total,
            "sentence": sentence,
            "actual": actual,
            "gemini": gemini_pred,
            "groq": groq_pred,
            "gemini_correct": gemini_ok,
            "groq_correct": groq_ok,
            "gemini_score": gemini_correct,
            "groq_score": groq_correct
        }

        if i < total - 1:
            time.sleep(delay_seconds)

