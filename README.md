# LLM Sarcasm Detection Benchmark

A Flask-based web platform to benchmark sarcasm detection performance of large language models using a labeled news headline dataset. The system compares Google Gemini and LLaMA-3.1 (via Groq) with live batch evaluation, real-time accuracy tracking, and interactive testing.

## Results
- Benchmarked **Gemini 2.5 Flash vs LLaMA-3.1** on **28,000+ sarcasm headlines**
- Real-time accuracy tracking via Server-Sent Events - live-updating scores without page refresh
- Prompt constraints (temperature=0, max 20 tokens) enforced for consistent, comparable outputs across both models

## Overview

This project provides an end-to-end evaluation framework for analyzing how different LLMs interpret sarcasm in short texts. It supports:

- Interactive single-sentence testing  
- Custom user-defined sentence evaluation  
- Dataset-level batch benchmarking  
- Real-time streaming of predictions and accuracy  
- Model-to-model comparison (Gemini vs LLaMA-3.1)

## Features

- LLM Integration
  - Google Gemini API
  - Groq API (LLaMA-3.1)

- Evaluation Modes
  - Single headline test
  - Custom sentence test
  - Live batch evaluation

- Metrics
  - Running accuracy per model
  - Final winner selection

## Dataset

[Sarcasm Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) - 28,000+ labeled news headlines. Dataset file included in the repository under `data/`.

## Run

pip install -r requirements.txt  
python app.py

Open: http://127.0.0.1:5000
