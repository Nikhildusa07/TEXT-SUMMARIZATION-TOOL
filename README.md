# Text Summarizer

Text Summarizer is a Flask-based NLP application that automatically generates concise summaries from large bodies of text using state-of-the-art **abstractive** and **extractive** techniques. It supports raw text, PDF, and DOCX inputs, and features a clean, interactive web interface.

---

## Features

- Paste text or upload PDF/DOCX files
- Choose between **Abstractive** or **Extractive** summarization
- Control summary length: short, medium, or long
- Real-time summarization using AJAX
- Download or copy the summary with one click
- Clean and responsive Flask UI

---

##  How It Works

The app leverages NLP models and summarization algorithms:

- **Abstractive Summarization**: Uses Hugging Face's T5 transformer model to generate human-like summaries.
- **Extractive Summarization**: Uses Sumy to extract key sentences directly from the input.

---

##  Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt



![Image](https://github.com/user-attachments/assets/48504796-47e4-4b8b-9831-34b9a9ab5bbe)
