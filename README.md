# LLM Sarcasm Comparison

A Flask app that compares Gemini 2.5 Flash-Lite and Llama 3.1 8B through Groq on sarcasm headline classification.

## What it does

The app supports single-headline, custom-text, and batch modes. Batch mode streams a live running score with Server-Sent Events.

The dataset is the 28K-headline Sarcasm Headlines set. A batch run scores a random sample of 10 headlines, as set by `BATCH_SIZE` in `app.py`, with a 5 second delay between calls. This is a comparison demo, not a full benchmark.

## Run locally

The dataset file is included in `data/`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

Enter your Gemini and Groq API keys in the app before running a model comparison. The batch endpoint uses POST so keys are not placed in the URL. Do not commit API keys.

## Topics

flask, sarcasm-detection, llm-evaluation, gemini, groq, server-sent-events
