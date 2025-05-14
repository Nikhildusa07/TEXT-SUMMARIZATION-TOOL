from transformers import pipeline

# Load the summarization model
print("Loading the summarization model...")
# Initialize the BART model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample input text (you can replace this with your own text)
input_text = """
The rapid advancement of artificial intelligence has transformed industries worldwide. From healthcare to finance, AI systems are being used to automate tasks, analyze data, and improve decision-making. In 2024, AI models became more efficient, with companies like xAI leading the charge in creating innovative solutions. However, challenges remain, such as ethical concerns around bias, transparency, and the environmental impact of training large models. Governments and organizations are now working together to establish regulations that ensure AI is used responsibly while fostering innovation.
"""

# Print the input text
print("\nInput Text:\n", input_text)

# Generate the summary
print("\nGenerating summary...")
# Summarize the input text using the BART model, with defined max and min length for the summary
summary = summarizer(input_text, max_length=75, min_length=25, do_sample=False)
summary_text = summary[0]['summary_text']

# Print the summary
print("\nSummary:\n", summary_text)
