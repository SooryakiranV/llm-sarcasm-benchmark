# LLM Sarcasm Detection Benchmark

A Flask-based web platform to benchmark sarcasm detection performance of large language models using a labeled news headline dataset. The system compares Google Gemini and LLaMA-3.1 (via Groq) with live batch evaluation, real-time accuracy tracking, and interactive testing.

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

Place the dataset file here:

data/Sarcasm_Headlines_Dataset.json

## Run

pip install -r requirements.txt  
python app.py

Open: http://127.0.0.1:5000
